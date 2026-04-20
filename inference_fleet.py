"""
Inference Script — Hiring Fleet: AI Oversight System
======================================================
Multi-agent fleet where a single LLM plays four distinct roles in sequence:
  1. Fraud Specialist  — hunts credential fraud and reference inconsistencies
  2. Skills Specialist — evaluates technical fit against job requirements
  3. Timeline Specialist — checks chronological consistency and career gaps
  4. Overseer          — synthesises specialist reports and issues final verdict

The environment controls phase transitions; the inference script simply responds
to the role_instructions in each observation with the appropriate action type.

MANDATORY VARIABLES:
    API_BASE_URL   LLM API endpoint.
    MODEL_NAME     Model identifier.
    HF_TOKEN       API key.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

import json
import os
import sys
import textwrap
import warnings
import logging
from typing import List, Optional

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

try:
    from dotenv import load_dotenv
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        load_dotenv()
except ImportError:
    pass

from openai import OpenAI

try:
    import requests
except ImportError:
    import urllib.request
    import urllib.error

    class _FallbackRequests:
        class _Response:
            def __init__(self, data, status_code):
                self.status_code = status_code
                self._data = data
            def json(self):
                return json.loads(self._data)
            def raise_for_status(self):
                if self.status_code >= 400:
                    raise Exception(f"HTTP {self.status_code}")

        @staticmethod
        def post(url, json=None, headers=None, timeout=None):
            import json as json_module
            data = json_module.dumps(json).encode("utf-8") if json else b""
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json", **(headers or {})},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout or 60) as resp:
                    body = resp.read().decode("utf-8")
                    return _FallbackRequests._Response(body, resp.status)
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8")
                return _FallbackRequests._Response(body, e.code)

        @staticmethod
        def get(url, timeout=None):
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout or 60) as resp:
                body = resp.read().decode("utf-8")
                return _FallbackRequests._Response(body, resp.status)

    requests = _FallbackRequests()


# ============================================================
# MANDATORY ENVIRONMENT VARIABLES
# ============================================================
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
ENV_URL = os.getenv("ENV_URL", "https://ishikamahadar-resume-env.hf.space")

BENCHMARK = "adversarial-resume-screening"
TASK_TYPES = ["easy", "medium", "hard"]
EPISODES_PER_TASK = 3
TEMPERATURE = 0.3
MAX_TOKENS = 700

SUCCESS_THRESHOLD = 0.4
MAX_REWARD_PER_EPISODE = 1.0
TOTAL_EPISODES = len(TASK_TYPES) * EPISODES_PER_TASK
MAX_TOTAL_REWARD = TOTAL_EPISODES * MAX_REWARD_PER_EPISODE


# ============================================================
# System prompts per role
# ============================================================
SYSTEM_PROMPTS = {
    "fraud_specialist": textwrap.dedent("""\
    You are a FRAUD SPECIALIST on a hiring review board. Your sole job is to detect fraud.
    Investigate the resume for fake credentials, inflated titles, false employers, or suspicious references.

    Available actions (output EXACTLY one JSON per turn):
    1. {"action_type":"view_section","section":"references"}
    2. {"action_type":"view_section","section":"header"}
    3. {"action_type":"check_reference","reference_id":"ref1"}
    4. {"action_type":"verify_credential"}
    5. {"action_type":"submit_specialist_report","findings":"<your findings>","has_issues":true|false,"specialist_confidence":0.0-1.0}

    IMPORTANT: When steps_remaining <= 1, you MUST call submit_specialist_report immediately.
    Respond with ONLY valid JSON, no other text.
    """),

    "skills_specialist": textwrap.dedent("""\
    You are a SKILLS SPECIALIST on a hiring review board. Your sole job is to assess technical fit.
    Compare the candidate's skills and experience against the job description requirements.

    Available actions (output EXACTLY one JSON per turn):
    1. {"action_type":"view_section","section":"experience"}
    2. {"action_type":"view_section","section":"education"}
    3. {"action_type":"view_section","section":"skills"}
    4. {"action_type":"view_section","section":"projects"}
    5. {"action_type":"ask_clarification","question":"<specific technical question>"}
    6. {"action_type":"submit_specialist_report","findings":"<your findings>","has_issues":true|false,"specialist_confidence":0.0-1.0}

    has_issues should be true if the candidate does NOT meet job requirements.
    IMPORTANT: When steps_remaining <= 1, you MUST call submit_specialist_report immediately.
    Respond with ONLY valid JSON, no other text.
    """),

    "timeline_specialist": textwrap.dedent("""\
    You are a TIMELINE SPECIALIST on a hiring review board. Your sole job is to check chronological consistency.
    Look for employment gaps, overlapping dates, or inconsistencies in the career timeline.

    Available actions (output EXACTLY one JSON per turn):
    1. {"action_type":"view_section","section":"header"}
    2. {"action_type":"view_section","section":"summary"}
    3. {"action_type":"view_section","section":"experience"}
    4. {"action_type":"ask_clarification","question":"<question about career gap or timeline>"}
    5. {"action_type":"submit_specialist_report","findings":"<your findings>","has_issues":true|false,"specialist_confidence":0.0-1.0}

    has_issues should be true if you found timeline inconsistencies, unexplained gaps, or overlapping employment.
    IMPORTANT: When steps_remaining <= 1, you MUST call submit_specialist_report immediately.
    Respond with ONLY valid JSON, no other text.
    """),

    "overseer": textwrap.dedent("""\
    You are the OVERSEER on a hiring review board. You receive reports from three specialist agents
    and make the final hiring decision.

    Available actions (output EXACTLY one JSON per turn):
    1. {"action_type":"request_reinvestigation","reinvestigation_target":"fraud_specialist|skills_specialist|timeline_specialist","reinvestigation_reason":"<why>"}
    2. {"action_type":"submit_final_decision","decision":"accept|reject","fraud_flag":true|false,"confidence":0.0-1.0,"fraud_reasoning":"<explanation if fraud>"}

    STRATEGY:
    - If specialist reports conflict or a fraud specialist flagged issues, consider request_reinvestigation (allowed once).
    - Otherwise, synthesise the reports and submit_final_decision immediately.
    - High confidence (0.8+) when all specialists agree. Lower confidence when reports conflict.
    IMPORTANT: When steps_remaining <= 1, you MUST call submit_final_decision immediately.
    Respond with ONLY valid JSON, no other text.
    """),

    "complete": textwrap.dedent("""\
    Episode complete. No actions needed.
    """),
}

FALLBACK_ACTIONS = {
    "fraud_specialist": {"action_type": "submit_specialist_report", "findings": "Unable to complete investigation.", "has_issues": False, "specialist_confidence": 0.3},
    "skills_specialist": {"action_type": "submit_specialist_report", "findings": "Unable to complete skills assessment.", "has_issues": False, "specialist_confidence": 0.3},
    "timeline_specialist": {"action_type": "submit_specialist_report", "findings": "Unable to complete timeline review.", "has_issues": False, "specialist_confidence": 0.3},
    "overseer": {"action_type": "submit_final_decision", "decision": "reject", "fraud_flag": False, "confidence": 0.4, "fraud_reasoning": ""},
}


# ============================================================
# HTTP Client
# ============================================================
class FleetHTTPClient:
    """HTTP client for the Fleet Resume Environment."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._current_episode_id: Optional[str] = None

    def reset(self, task_type: str = "easy", seed: int = 42) -> dict:
        self._current_episode_id = f"fleet-{task_type}-{seed}"
        payload = {
            "episode_id": self._current_episode_id,
            "seed": seed,
            "task_type": task_type,
        }
        resp = requests.post(f"{self.base_url}/reset", json=payload, timeout=30)
        resp.raise_for_status()
        return self._parse(resp.json(), task_type)

    def step(self, action: dict) -> dict:
        clean_action = {k: v for k, v in action.items() if v is not None}
        if self._current_episode_id:
            clean_action["episode_id"] = self._current_episode_id
        payload = {"action": clean_action, "timeout_s": 30}
        resp = requests.post(f"{self.base_url}/step", json=payload, timeout=30)
        resp.raise_for_status()
        return self._parse(resp.json())

    @staticmethod
    def _parse(data: dict, fallback_task: str = "easy") -> dict:
        obs = data.get("observation", data)
        return {
            "task_type": obs.get("task_type", fallback_task),
            "current_phase": obs.get("current_phase", "fraud_specialist"),
            "role_instructions": obs.get("role_instructions", ""),
            "job_description": obs.get("job_description", ""),
            "visible_sections": obs.get("visible_sections", {}),
            "specialist_reports": obs.get("specialist_reports", []),
            "available_actions": obs.get("available_actions", []),
            "clarification_response": obs.get("clarification_response"),
            "reference_response": obs.get("reference_response"),
            "verification_result": obs.get("verification_result"),
            "steps_remaining": obs.get("steps_remaining", 0),
            "total_steps_remaining": obs.get("total_steps_remaining", 0),
            "feedback": obs.get("feedback", ""),
            "done": data.get("done", obs.get("done", False)),
            "reward": data.get("reward", obs.get("reward", 0.0)) or 0.0,
        }


# ============================================================
# Prompt builder
# ============================================================
def build_user_prompt(obs: dict, step: int, history: List[str]) -> str:
    phase = obs.get("current_phase", "fraud_specialist")
    visible = obs.get("visible_sections", {})
    sections_text = "".join(
        f"\n--- {name.upper()} ---\n{content}\n"
        for name, content in visible.items()
    )

    reports_text = ""
    for r in obs.get("specialist_reports", []):
        role = r.get("specialist_role", "?")
        findings = r.get("findings", "")
        has_issues = r.get("has_issues", False)
        conf = r.get("confidence", 0.0)
        reports_text += (
            f"\n[{role.upper()} REPORT]\n"
            f"  Has Issues: {has_issues} | Confidence: {conf:.2f}\n"
            f"  Findings: {findings}\n"
        )

    extra = ""
    if obs.get("clarification_response"):
        extra += f"\nCandidate clarification: {obs['clarification_response']}"
    if obs.get("reference_response"):
        extra += f"\nReference check result: {obs['reference_response']}"
    if obs.get("verification_result"):
        extra += f"\nCredential verification: {obs['verification_result']}"

    history_text = "\n".join(history[-4:]) if history else "None"
    return textwrap.dedent(f"""\
Step {step} | Phase: {phase} | Steps remaining in phase: {obs.get('steps_remaining', 0)} | Total steps left: {obs.get('total_steps_remaining', 0)}
Feedback: {obs.get('feedback', '')}

Job Description:
{obs.get('job_description', '')}

Revealed Resume Sections:{sections_text}
{f"Specialist Reports:{reports_text}" if reports_text else ""}
{extra}

Previous actions this episode:
{history_text}

Available actions: {obs.get('available_actions', [])}

Choose your next action (JSON only).
{"URGENT: steps_remaining <= 1 — you MUST submit your report/decision NOW." if obs.get('steps_remaining', 0) <= 1 else ""}
""")


# ============================================================
# LLM action parser
# ============================================================
def parse_action(client: OpenAI, obs: dict, step: int, history: List[str]) -> dict:
    phase = obs.get("current_phase", "fraud_specialist")
    system_prompt = SYSTEM_PROMPTS.get(phase, SYSTEM_PROMPTS["overseer"])
    fallback = FALLBACK_ACTIONS.get(phase, FALLBACK_ACTIONS["overseer"])

    # Force terminal action if budget exhausted
    if obs.get("steps_remaining", 0) <= 1:
        return fallback

    user_prompt = build_user_prompt(obs, step, history)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            response_format={"type": "json_object"},
            stream=False,
        )
        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)

        if "action_type" not in parsed:
            return fallback

        action = _build_action(parsed, phase, fallback)
        return action

    except Exception:
        return fallback


def _build_action(parsed: dict, phase: str, fallback: dict) -> dict:
    at = parsed.get("action_type", "")

    if at == "view_section":
        return {"action_type": "view_section", "section": parsed.get("section", "experience")}

    elif at == "ask_clarification":
        return {"action_type": "ask_clarification", "question": parsed.get("question", "Please elaborate.")}

    elif at == "check_reference":
        return {"action_type": "check_reference", "reference_id": parsed.get("reference_id", "ref1")}

    elif at == "verify_credential":
        return {"action_type": "verify_credential"}

    elif at == "submit_specialist_report":
        return {
            "action_type": "submit_specialist_report",
            "findings": parsed.get("findings", "No specific findings."),
            "has_issues": bool(parsed.get("has_issues", False)),
            "specialist_confidence": float(parsed.get("specialist_confidence", 0.5)),
        }

    elif at == "request_reinvestigation":
        return {
            "action_type": "request_reinvestigation",
            "reinvestigation_target": parsed.get("reinvestigation_target", "fraud_specialist"),
            "reinvestigation_reason": parsed.get("reinvestigation_reason", "Need more information."),
        }

    elif at == "submit_final_decision":
        return {
            "action_type": "submit_final_decision",
            "decision": parsed.get("decision", "reject"),
            "fraud_flag": bool(parsed.get("fraud_flag", False)),
            "confidence": float(parsed.get("confidence", 0.5)),
            "fraud_reasoning": parsed.get("fraud_reasoning", ""),
        }

    return fallback


def action_to_str(action: dict) -> str:
    at = action.get("action_type", "unknown")
    if at == "view_section":
        return f"view_section({action.get('section', '')})"
    elif at == "ask_clarification":
        return f"ask_clarification({str(action.get('question', ''))[:35]})"
    elif at == "check_reference":
        return f"check_reference({action.get('reference_id', '')})"
    elif at == "verify_credential":
        return "verify_credential()"
    elif at == "submit_specialist_report":
        return f"submit_specialist_report(issues={action.get('has_issues', False)},conf={action.get('specialist_confidence', 0):.2f})"
    elif at == "request_reinvestigation":
        return f"request_reinvestigation({action.get('reinvestigation_target', '')})"
    elif at == "submit_final_decision":
        return f"submit_final_decision({action.get('decision', '')},fraud={action.get('fraud_flag', False)},conf={action.get('confidence', 0):.2f})"
    return f"{at}()"


# ============================================================
# Logging helpers
# ============================================================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ============================================================
# Episode runner
# ============================================================
def run_episode(client: OpenAI, env: FleetHTTPClient, task_type: str, episode_num: int) -> tuple:
    obs = env.reset(task_type=task_type, seed=episode_num)

    history: List[str] = []
    episode_rewards: List[float] = []
    step = 0
    task_name = f"fleet-{task_type}-{episode_num}"

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        while True:
            step += 1
            if obs.get("done", False):
                break

            action_dict = parse_action(client, obs, step, history)
            action_str = action_to_str(action_dict)

            obs = env.step(action_dict)
            reward = obs.get("reward", 0.0) or 0.0
            done = obs.get("done", False)

            episode_rewards.append(reward)
            log_step(step=step, action=action_str, reward=reward, done=done, error=None)
            history.append(f"Step {step} [{obs.get('current_phase', '?')}]: {action_str} -> reward={reward:+.3f}")

            if done:
                break
            if step >= 20:  # safety cap
                break

    except Exception as e:
        episode_rewards.append(0.0)
        log_step(step=step, action="error", reward=0.0, done=True, error=str(e))

    total = sum(episode_rewards)
    score = max(0.0, min(1.0, total / MAX_REWARD_PER_EPISODE)) if MAX_REWARD_PER_EPISODE > 0 else 0.0
    success = score >= SUCCESS_THRESHOLD
    log_end(success=success, steps=step, score=score, rewards=episode_rewards)
    return step, episode_rewards


# ============================================================
# Main
# ============================================================
def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = FleetHTTPClient(base_url=ENV_URL)

    all_rewards: List[float] = []
    total_steps = 0

    for task_type in TASK_TYPES:
        for ep in range(1, EPISODES_PER_TASK + 1):
            steps, rewards = run_episode(client, env, task_type, ep)
            total_steps += steps
            all_rewards.extend(rewards)

    overall_score = sum(all_rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
    overall_score = max(0.0, min(1.0, overall_score))


if __name__ == "__main__":
    main()
