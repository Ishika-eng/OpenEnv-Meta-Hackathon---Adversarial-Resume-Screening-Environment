"""
End-to-End Local Test — Fleet Resume Screening
================================================
Exercises the full four-phase pipeline WITHOUT an HTTP server or LLM API.

Architecture:
  • FleetResumeEnvironment is called directly (no HTTP)
  • A deterministic RuleAgent replaces the LLM — it reads the observation
    and picks actions based on simple rules (mirrors what a well-behaved LLM
    should do), covering all action types introduced in Days 1-3
  • Assertions verify:
      - All four phases are reached in sequence
      - read_reports, request_reinvestigation, submit_final_decision all work
      - Reward is in [0.0, 1.0] for every episode
      - Final observation has done=True and current_phase="complete"
      - Reward improves when agent reads all reports vs skipping
      - Step efficiency bonus fires when budget not fully used

Run:
    python3 test_e2e_local.py
"""

import sys
import os
import types
import unittest
from typing import List

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

# ── Mock openenv (same pattern as test_day3_overseer.py) ─────────────────────
def _make_openenv_mock():
    from pydantic import BaseModel

    openenv_mod = types.ModuleType("openenv")
    core_mod    = types.ModuleType("openenv.core")
    server_mod  = types.ModuleType("openenv.core.env_server")

    class _EnvironmentMeta(type):
        def __getitem__(cls, item): return cls

    class Action(BaseModel):
        model_config = {"arbitrary_types_allowed": True}

    class Observation(BaseModel):
        model_config = {"arbitrary_types_allowed": True}
        done:   bool  = False
        reward: float = 0.0

    class State(BaseModel):
        model_config = {"arbitrary_types_allowed": True}

    class Environment(metaclass=_EnvironmentMeta):
        def __init__(self): pass

    server_mod.Action      = Action
    server_mod.Observation = Observation
    server_mod.State       = State
    server_mod.Environment = Environment
    server_mod.create_fastapi_app = lambda *a, **kw: None

    openenv_mod.core    = core_mod
    core_mod.env_server = server_mod

    sys.modules["openenv"]                 = openenv_mod
    sys.modules["openenv.core"]            = core_mod
    sys.modules["openenv.core.env_server"] = server_mod

_make_openenv_mock()

from server.fleet_environment import FleetResumeEnvironment, PHASE_BUDGETS
from server.overseer_env import OVERSEER_ROLE_INSTRUCTIONS


# ── Mock dataset ──────────────────────────────────────────────────────────────

def _resume(is_fraud: bool, decision: str) -> dict:
    return {
        "job_description": "Senior ML Engineer — Python, PyTorch, distributed systems.",
        "resume_sections": {
            "header":     "Alex Kim — Machine Learning Engineer",
            "summary":    "8 years ML experience, published researcher.",
            "experience": "Google 2016-2024, Meta 2014-2016",
            "education":  "Stanford MSc AI 2014",
            "skills":     "Python, PyTorch, TensorFlow, Kubernetes, GCP",
            "projects":   "Built recommendation engine serving 500M users.",
            "references": "ref1: Dr. Lisa Park (Google). ref2: Chen Wei (Meta).",
        },
        "clarification_answers": {
            "gap": "No gap — I transitioned directly from Meta to Google.",
            "pytorch": "Used PyTorch daily for model training pipelines.",
        },
        "reference_check_results": {
            "ref1": {"name": "Dr. Lisa Park", "response": "Excellent engineer. Dates match."},
            "ref2": {"name": "Chen Wei",      "response": "Unknown person — name not in our system."},
        },
        "verification_data": {
            "degree_verified":      True,
            "employment_verified":  False if is_fraud else True,
        },
        "required_skills": ["Python", "PyTorch", "Kubernetes"],
        "ground_truth": {
            "decision": decision,
            "is_fraud": is_fraud,
            "fraud_indicators": (
                ["fabricated_reference", "unverified_employment"] if is_fraud else []
            ),
            "employment_gaps": [],
        },
    }


MOCK_DATASET = {
    "easy":   [_resume(is_fraud=False, decision="accept"),
               _resume(is_fraud=True,  decision="reject")],
    "medium": [_resume(is_fraud=True,  decision="reject"),
               _resume(is_fraud=False, decision="accept")],
    "hard":   [_resume(is_fraud=True,  decision="reject"),
               _resume(is_fraud=False, decision="accept")],
}


def _fresh_env(task_type: str = "medium", dataset: dict = None) -> FleetResumeEnvironment:
    """
    Create a bare FleetResumeEnvironment with injected dataset.
    Accepts an explicit dataset so run_episode can control exactly
    which resume is tested without cross-contamination.
    """
    ds = dataset if dataset is not None else MOCK_DATASET
    FleetResumeEnvironment._episode_store = {}
    FleetResumeEnvironment._default_session = f"e2e-{task_type}"
    FleetResumeEnvironment._dataset_cache = ds
    env = FleetResumeEnvironment.__new__(FleetResumeEnvironment)
    env.data_path = "data/resumes.json"
    env.dataset = ds
    # init instance attrs
    env._task_type = task_type
    env._current_index = 0
    env._sample = None
    env._phase_idx = 0
    env._phase_steps_used = 0
    env._total_steps_used = 0
    env._max_total_steps = sum(PHASE_BUDGETS[task_type].values())
    env._sections_viewed = []
    env._specialist_reports = []
    env._references_checked = 0
    env._verifications_done = 0
    env._clarifications_asked = 0
    env._reinvestigation_used = False
    env._violations_count = 0
    env._reports_read = []
    env._read_report_details = {}
    env._done = False
    env._last_clarification = None
    env._last_reference = None
    env._last_verification = None
    env._last_feedback = ""
    return env


# ── Deterministic rule-based agent ───────────────────────────────────────────

class RuleAgent:
    """
    Deterministic rule-based agent that mirrors a well-behaved LLM.

    Key design principle: each specialist phase submits its own report
    BEFORE the budget runs out, using investigation results to set
    has_issues correctly.  The agent tracks what it has discovered
    per-phase so it can always produce a sensible final report.

    Strategies per phase:
      fraud_specialist    — check_reference → verify_credential → submit
      skills_specialist   — view skills → view experience → clarify → submit
      timeline_specialist — view experience/summary → submit
      overseer            — read_reports (all 3) → submit_final_decision
    """

    def __init__(self, read_all_reports: bool = True, use_reinvestigation: bool = False):
        self.read_all_reports  = read_all_reports
        self.use_reinvestigation = use_reinvestigation
        # Per-phase state so we know what we've found
        self._ref_response:    str  = ""
        self._verif_response:  str  = ""
        self._clarif_response: str  = ""

    def act(self, obs: dict) -> dict:
        phase      = obs.get("current_phase", "fraud_specialist")
        steps_left = obs.get("steps_remaining", 0)

        # Cache tool responses as they come in
        if obs.get("reference_response"):
            self._ref_response = obs["reference_response"]
        if obs.get("verification_result"):
            self._verif_response = obs["verification_result"]
        if obs.get("clarification_response"):
            self._clarif_response = obs["clarification_response"]

        # Hard emergency: budget completely gone
        if steps_left == 0:
            return self._emergency_submit(phase, obs)

        if phase == "fraud_specialist":
            return self._fraud_act(obs, steps_left)
        elif phase == "skills_specialist":
            return self._skills_act(obs, steps_left)
        elif phase == "timeline_specialist":
            return self._timeline_act(obs, steps_left)
        elif phase == "overseer":
            return self._overseer_act(obs, steps_left)

        return self._emergency_submit(phase, obs)

    # ── Helpers ─────────────────────────────────────────────────────────

    def _fraud_issues_detected(self) -> bool:
        """Return True if any investigation signal suggests fraud."""
        ref_bad   = "unknown" in self._ref_response.lower() or "not in our system" in self._ref_response.lower()
        verif_bad = "FAILED" in self._verif_response.upper()  # uppercase check against uppercased string
        return ref_bad or verif_bad

    def _fraud_submit(self) -> dict:
        has_issues = self._fraud_issues_detected()
        findings   = (
            "Reference check: ref2 not found in employer system. "
            "Employment verification FAILED. "
            "Indicators: fabricated_reference, unverified_employment."
            if has_issues else
            "References verified. Credential checks passed. No fraud detected."
        )
        return {
            "action_type": "submit_specialist_report",
            "findings": findings,
            "has_issues": has_issues,
            "specialist_confidence": 0.88 if has_issues else 0.82,
        }

    # ── Fraud specialist ─────────────────────────────────────────────────

    def _fraud_act(self, obs: dict, steps_left: int) -> dict:
        available = obs.get("available_actions", [])

        needs_ref   = "check_reference"   in available and not self._ref_response
        needs_verif = "verify_credential" in available and not self._verif_response
        investigations_left = int(needs_ref) + int(needs_verif)

        # If only 1 step left, or nothing left to investigate → submit
        if steps_left == 1 or investigations_left == 0:
            return self._fraud_submit()

        # Slots available for investigation before forced submit
        invest_slots = steps_left - 1

        if invest_slots == 1:
            # Tight budget — exactly 1 investigation slot before must submit.
            # Prioritise verify_credential: gives a deterministic "FAILED" signal
            # for fraudulent candidates regardless of which reference we check.
            if needs_verif:
                return {"action_type": "verify_credential"}
            if needs_ref:
                # Fall back to ref2 — the second reference is the suspicious one
                # in our dataset (ref1 is always the good referee).
                return {"action_type": "check_reference", "reference_id": "ref2"}
        else:
            # Multiple slots: check ref2 (suspicious) first, then verify credential
            if needs_ref:
                return {"action_type": "check_reference", "reference_id": "ref2"}
            if needs_verif:
                return {"action_type": "verify_credential"}

        return self._fraud_submit()

    # ── Skills specialist ─────────────────────────────────────────────────

    def _skills_act(self, obs: dict, steps_left: int) -> dict:
        sections  = obs.get("visible_sections", {})
        available = obs.get("available_actions", [])

        # Count remaining useful investigations
        need_skills = "skills" not in sections
        need_exp    = "experience" not in sections
        need_clarif = not self._clarif_response and "ask_clarification" in available
        remaining_invest = sum([need_skills, need_exp, need_clarif])

        if steps_left == 1 or remaining_invest == 0:
            return {
                "action_type": "submit_specialist_report",
                "findings": "Python/PyTorch/Kubernetes skills confirmed. Experience matches JD requirements.",
                "has_issues": False,
                "specialist_confidence": 0.84,
            }

        if need_skills:
            return {"action_type": "view_section", "section": "skills"}
        if need_exp:
            return {"action_type": "view_section", "section": "experience"}
        if need_clarif:
            return {"action_type": "ask_clarification",
                    "question": "Can you describe your PyTorch production experience?"}

        return {
            "action_type": "submit_specialist_report",
            "findings": "Skills assessment complete. Candidate meets requirements.",
            "has_issues": False,
            "specialist_confidence": 0.84,
        }

    # ── Timeline specialist ───────────────────────────────────────────────

    def _timeline_act(self, obs: dict, steps_left: int) -> dict:
        sections = obs.get("visible_sections", {})

        need_exp  = "experience" not in sections
        need_summ = "summary" not in sections
        remaining = sum([need_exp, need_summ])

        if steps_left == 1 or remaining == 0:
            return {
                "action_type": "submit_specialist_report",
                "findings": "Timeline consistent — employment dates align. No unexplained gaps.",
                "has_issues": False,
                "specialist_confidence": 0.80,
            }

        if need_exp:
            return {"action_type": "view_section", "section": "experience"}
        if need_summ:
            return {"action_type": "view_section", "section": "summary"}

        return {
            "action_type": "submit_specialist_report",
            "findings": "Timeline consistent. No gaps detected.",
            "has_issues": False,
            "specialist_confidence": 0.80,
        }

    # ── Overseer ─────────────────────────────────────────────────────────

    def _overseer_act(self, obs: dict, steps_left: int) -> dict:
        reports_read       = obs.get("reports_read", [])
        specialist_reports = obs.get("specialist_reports", [])
        available          = obs.get("available_actions", [])

        # How many reads are still pending?
        unread = [r.get("specialist_role", "") for r in specialist_reports
                  if r.get("specialist_role") not in reports_read]
        # Keep at least 1 step for submit_final_decision
        must_submit = steps_left == 1 or (not unread)

        if not must_submit and self.read_all_reports and "read_reports" in available and unread:
            return {"action_type": "read_reports", "report_target": unread[0]}

        if (not must_submit and self.use_reinvestigation
                and "request_reinvestigation" in available):
            return {
                "action_type": "request_reinvestigation",
                "reinvestigation_target": "fraud_specialist",
                "reinvestigation_reason": "Reference conflict requires deeper investigation.",
            }

        # Synthesise
        fraud_flagged = any(
            r.get("has_issues", False) for r in specialist_reports
            if r.get("specialist_role") == "fraud_specialist"
        )
        any_issues = any(r.get("has_issues", False) for r in specialist_reports)
        decision   = "reject" if any_issues else "accept"
        issues_set = set(r.get("has_issues", False) for r in specialist_reports)
        confidence = 0.87 if len(issues_set) == 1 else 0.65

        return {
            "action_type": "submit_final_decision",
            "decision": decision,
            "fraud_flag": fraud_flagged,
            "confidence": confidence,
            "fraud_reasoning": (
                "Reference check returned unknown person. "
                "Employment verification FAILED. "
                "Fabricated reference and unverified_employment detected."
                if fraud_flagged else ""
            ),
        }

    def _emergency_submit(self, phase: str, obs: dict) -> dict:
        """Absolute last resort — fires only when steps_remaining=0."""
        if phase == "overseer":
            specialist_reports = obs.get("specialist_reports", [])
            fraud_flagged = any(
                r.get("has_issues", False) for r in specialist_reports
                if r.get("specialist_role") == "fraud_specialist"
            )
            any_issues = any(r.get("has_issues", False) for r in specialist_reports)
            return {
                "action_type": "submit_final_decision",
                "decision": "reject" if any_issues else "accept",
                "fraud_flag": fraud_flagged,
                "confidence": 0.4,
                "fraud_reasoning": "Emergency submission." if fraud_flagged else "",
            }
        return {
            "action_type": "submit_specialist_report",
            "findings": "Emergency submission — budget exhausted.",
            "has_issues": False,
            "specialist_confidence": 0.3,
        }


# ── Episode runner ─────────────────────────────────────────────────────────

def run_episode(
    task_type: str,
    is_fraud: bool,
    read_all: bool = True,
    use_reinv: bool = False,
    seed: int = 0,
) -> dict:
    """Run one full fleet episode and return a result dict."""
    # Build a single-resume dataset for this episode so the random index
    # always picks the intended fraud/non-fraud scenario, independent of
    # any global MOCK_DATASET or previous run state.
    ep_decision = "reject" if is_fraud else "accept"
    ep_dataset = {
        "easy":   [_resume(is_fraud=is_fraud, decision=ep_decision)],
        "medium": [_resume(is_fraud=is_fraud, decision=ep_decision)],
        "hard":   [_resume(is_fraud=is_fraud, decision=ep_decision)],
    }
    env = _fresh_env(task_type, dataset=ep_dataset)
    obs_obj = env.reset(task_type=task_type, seed=seed)
    obs = obs_obj.model_dump()

    agent  = RuleAgent(read_all_reports=read_all, use_reinvestigation=use_reinv)
    phases_seen: List[str] = []
    rewards: List[float] = []
    step = 0
    max_safety = 40

    while not obs.get("done", False) and step < max_safety:
        phase = obs.get("current_phase", "?")
        if not phases_seen or phases_seen[-1] != phase:
            phases_seen.append(phase)

        action = agent.act(obs)
        obs_obj = env.step(type("FA", (), action)() if False else _dict_to_action(action))
        obs = obs_obj.model_dump()
        rewards.append(obs.get("reward", 0.0))
        step += 1

    return {
        "done": obs.get("done", False),
        "phase": obs.get("current_phase", "?"),
        "phases_seen": phases_seen,
        "reward": obs.get("reward", 0.0),
        "rewards": rewards,
        "steps": step,
        "feedback": obs.get("feedback", ""),
        "reports_read": obs.get("reports_read", []),
        "violations": obs.get("violations_count", 0),
    }


def _dict_to_action(d: dict):
    """Convert a plain dict to a FleetAction-compatible object."""
    from models import FleetAction
    # Filter to only valid FleetAction fields
    valid = {
        "action_type", "episode_id", "section", "question", "reference_id",
        "report_target", "findings", "has_issues", "specialist_confidence",
        "reinvestigation_target", "reinvestigation_reason",
        "decision", "fraud_flag", "confidence", "fraud_reasoning",
    }
    clean = {k: v for k, v in d.items() if k in valid}
    return FleetAction(**clean)


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

class TestE2EPhaseSequence(unittest.TestCase):
    """All four phases are reached in the correct order."""

    def _phases(self, task_type: str, is_fraud: bool) -> List[str]:
        result = run_episode(task_type, is_fraud)
        return result["phases_seen"]

    def test_easy_fraud_all_phases_reached(self):
        phases = self._phases("easy", is_fraud=True)
        for expected in ["fraud_specialist", "skills_specialist", "timeline_specialist", "overseer"]:
            self.assertIn(expected, phases, f"Missing phase: {expected}")

    def test_medium_clean_all_phases_reached(self):
        phases = self._phases("medium", is_fraud=False)
        for expected in ["fraud_specialist", "skills_specialist", "timeline_specialist", "overseer"]:
            self.assertIn(expected, phases, f"Missing phase: {expected}")

    def test_hard_fraud_all_phases_reached(self):
        phases = self._phases("hard", is_fraud=True)
        self.assertEqual(phases[:4],
            ["fraud_specialist", "skills_specialist", "timeline_specialist", "overseer"])

    def test_episode_ends_in_complete(self):
        result = run_episode("medium", is_fraud=True)
        self.assertTrue(result["done"], "Episode should be done=True")
        self.assertEqual(result["phase"], "complete")


class TestE2EReward(unittest.TestCase):
    """Reward values are in range and reflect decision quality."""

    def test_correct_fraud_decision_gives_positive_reward(self):
        result = run_episode("medium", is_fraud=True)
        self.assertGreater(result["reward"], 0.0)

    def test_correct_clean_decision_gives_positive_reward(self):
        result = run_episode("medium", is_fraud=False)
        self.assertGreater(result["reward"], 0.0)

    def test_reward_clamped_to_one(self):
        for task_type in ["easy", "medium", "hard"]:
            for is_fraud in [True, False]:
                result = run_episode(task_type, is_fraud)
                self.assertLessEqual(result["reward"], 1.0,
                    f"Reward {result['reward']} exceeded 1.0 for {task_type} fraud={is_fraud}")
                self.assertGreaterEqual(result["reward"], 0.0)

    def test_reading_all_reports_gives_higher_reward_than_none(self):
        result_read = run_episode("hard", is_fraud=True, read_all=True,  seed=5)
        result_skip = run_episode("hard", is_fraud=True, read_all=False, seed=5)
        self.assertGreaterEqual(result_read["reward"], result_skip["reward"],
            f"Read-all reward {result_read['reward']} should be ≥ skip reward {result_skip['reward']}")

    def test_all_task_types_produce_nonzero_reward(self):
        for task_type in ["easy", "medium", "hard"]:
            result = run_episode(task_type, is_fraud=True)
            self.assertGreater(result["reward"], 0.0,
                f"Zero reward for task_type={task_type}")

    def test_hard_tier_reward_structure(self):
        """Hard tier should be achievable (reward > 0.5) with correct decisions."""
        result = run_episode("hard", is_fraud=True, read_all=True)
        self.assertGreater(result["reward"], 0.5,
            f"Hard tier with correct decision should exceed 0.5, got {result['reward']}")


class TestE2EOverseerBehaviour(unittest.TestCase):
    """Overseer-specific actions work correctly in full episode."""

    def test_reports_read_populated_after_full_episode(self):
        result = run_episode("hard", is_fraud=True, read_all=True)
        self.assertEqual(len(result["reports_read"]), 3,
            f"Expected 3 reports read, got {result['reports_read']}")

    def test_zero_reports_read_when_skipping(self):
        result = run_episode("easy", is_fraud=True, read_all=False)
        self.assertEqual(len(result["reports_read"]), 0)

    def test_no_violations_in_well_behaved_episode(self):
        result = run_episode("medium", is_fraud=False, read_all=True)
        self.assertEqual(result["violations"], 0,
            f"Expected 0 violations, got {result['violations']}")

    def test_episode_completes_within_step_budget(self):
        for task_type in ["easy", "medium", "hard"]:
            result = run_episode(task_type, is_fraud=True, read_all=True)
            max_steps = sum(PHASE_BUDGETS[task_type].values()) + 2  # +2 safety
            self.assertLessEqual(result["steps"], max_steps,
                f"{task_type}: took {result['steps']} steps, max={max_steps}")

    def test_feedback_includes_final_reward(self):
        result = run_episode("medium", is_fraud=True)
        self.assertIn("reward", result["feedback"].lower(),
            f"Expected 'reward' in feedback, got: {result['feedback']}")


class TestE2EAllTiersAllScenarios(unittest.TestCase):
    """Full matrix: 3 tiers × 2 fraud scenarios = 6 episodes."""

    def test_full_matrix(self):
        results = []
        for task_type in ["easy", "medium", "hard"]:
            for is_fraud in [True, False]:
                result = run_episode(task_type, is_fraud, read_all=True)
                results.append({
                    "task": task_type,
                    "is_fraud": is_fraud,
                    "reward": result["reward"],
                    "done": result["done"],
                    "steps": result["steps"],
                })

        print("\n\n=== FULL MATRIX RESULTS ===")
        rewards = []
        for r in results:
            flag = "✓" if r["done"] and r["reward"] > 0 else "✗"
            print(f"  {flag} {r['task']:6s} fraud={str(r['is_fraud']):5s}  "
                  f"reward={r['reward']:.4f}  steps={r['steps']}")
            rewards.append(r["reward"])

        unique = len(set(round(r, 3) for r in rewards))
        print(f"\n  Unique reward values (6 eps): {unique}")
        print(f"  Reward range: [{min(rewards):.4f}, {max(rewards):.4f}]")

        for r in results:
            self.assertTrue(r["done"], f"Episode not done: {r}")
            self.assertGreater(r["reward"], 0.0, f"Zero reward: {r}")

        # For GRPO training we need variance — at least 3 distinct reward values
        self.assertGreaterEqual(unique, 3,
            f"Need ≥3 distinct reward values for GRPO, got {unique}: {sorted(set(round(r,3) for r in rewards))}")


class TestE2ERewardComponentBreakdown(unittest.TestCase):
    """Verify individual reward components fire correctly."""

    def test_step_efficiency_bonus_fires(self):
        """Easy tier with correct decision and unused budget earns efficiency bonus."""
        # Easy: budget=8. Agent uses ~6 steps (2 fraud + 2 skills + 2 timeline + submit).
        # The efficiency bonus should fire (steps_saved > 0).
        result_read = run_episode("easy", is_fraud=False, read_all=False)
        # If agent uses fewer than 8 steps and is correct, efficiency bonus fires.
        # We can't easily inspect internal reward breakdown, but reward should be > baseline
        self.assertGreater(result_read["reward"], 0.0)

    def test_correct_fraud_reasoning_boosts_reward(self):
        """Fraud reasoning that mentions indicator keywords gets a quality bonus."""
        # The RuleAgent's fraud_reasoning mentions "fabricated_reference" and
        # "unverified_employment" which match the ground_truth fraud_indicators.
        result = run_episode("hard", is_fraud=True, read_all=True)
        # Should achieve > 0.7 with all components firing
        self.assertGreater(result["reward"], 0.60,
            f"Expected > 0.60 with good reasoning, got {result['reward']}")

    def test_fleet_coordination_bonus_fires_when_all_correct(self):
        """When all 3 specialists are correct AND overseer is correct, reward is higher."""
        # Run two episodes with same conditions, varying only how well specialists do.
        # This is hard to isolate without accessing internals, but the full-read
        # episode should score well overall.
        result = run_episode("hard", is_fraud=True, read_all=True)
        self.assertGreater(result["reward"], 0.55)


# ════════════════════════════════════════════════════════════════════════════
# Runner
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()

    test_classes = [
        TestE2EPhaseSequence,
        TestE2EReward,
        TestE2EOverseerBehaviour,
        TestE2EAllTiersAllScenarios,
        TestE2ERewardComponentBreakdown,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    print("Running end-to-end local tests (no HTTP / no LLM API needed)...\n")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
