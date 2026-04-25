"""
train_grpo.py — GRPO training for the Hiring Fleet environment.

HOW IT WORKS:
  Step 1 (offline): Use a rule-based agent to walk through the environment
                    and collect diverse observations (one per phase per episode).
                    No model/GPU needed for this step.

  Step 2 (training): GRPOTrainer generates 8 completions per prompt internally,
                     calls our reward_fn to score each one, then updates the model.
                     The reward_fn does NOT call the live environment — it scores
                     the completion directly (valid JSON, correct phase action, etc.)

This is the correct GRPOTrainer workflow: dataset = prompts only, reward_fn scores
completions. The trainer handles generation internally.

Usage on Kaggle (T4 GPU, 16 GB):
    Set ENV_URL in the cell above, then: main()
"""

import os, json, re, random, time, traceback
from typing import Optional

import torch
import requests
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset

# ─────────────────────────────────────────────────────────────────────────────
# Config  — edit these before running
# ─────────────────────────────────────────────────────────────────────────────
ENV_URL    = os.getenv("ENV_URL",    "https://ishikamahadar-resume-env.hf.space")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")
HF_TOKEN   = os.getenv("HF_TOKEN",  "")      # only needed for gated models

N_COLLECT_EPISODES = 36    # episodes to walk through for prompt collection
GROUP_SIZE         = 4     # completions per prompt (lower = less VRAM)
OUTPUT_DIR         = "grpo_fleet_output"

PHASE_ORDER = ["fraud_specialist", "skills_specialist", "timeline_specialist", "overseer"]

# Optimal action for each phase — used in the reward function
PHASE_BEST_ACTIONS = {
    "fraud_specialist":   ["verify_credential", "check_reference"],
    "skills_specialist":  ["view_section", "ask_clarification"],
    "timeline_specialist":["view_section", "ask_clarification"],
    "overseer":           ["read_reports", "submit_final_decision"],
}

# Sections worth viewing (high-value)
HIGH_VALUE_SECTIONS = {"experience", "education", "skills"}

# ─────────────────────────────────────────────────────────────────────────────
# System prompt  (same at training and inference)
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a hiring fleet agent. Based on the observation below, output a single JSON \
action object and nothing else.

Valid action types:
  fraud_specialist  → verify_credential | check_reference | view_section | submit_specialist_report
  skills_specialist → view_section | ask_clarification | submit_specialist_report
  timeline_specialist → view_section | ask_clarification | submit_specialist_report
  overseer          → read_reports | request_reinvestigation | submit_final_decision

Examples:
  {"action_type": "verify_credential"}
  {"action_type": "check_reference", "reference_id": "ref2"}
  {"action_type": "view_section", "section": "experience"}
  {"action_type": "ask_clarification", "question": "When exactly did you join TechCorp?"}
  {"action_type": "submit_specialist_report", "findings": "Credential FAILED. Reference denies employment.", "has_issues": true, "specialist_confidence": 0.9}
  {"action_type": "read_reports", "report_target": "fraud_specialist"}
  {"action_type": "submit_final_decision", "decision": "reject", "fraud_flag": true, "confidence": 0.88, "fraud_reasoning": "Credential FAILED, reference denied employment."}

Output ONLY the JSON object. No explanation."""


# ─────────────────────────────────────────────────────────────────────────────
# Observation → prompt string
# ─────────────────────────────────────────────────────────────────────────────
def obs_to_prompt(obs: dict) -> str:
    lines = [
        f"PHASE: {obs.get('current_phase', '?')}",
        f"STEPS REMAINING: {obs.get('steps_remaining', 0)}  "
        f"(total left: {obs.get('total_steps_remaining', 0)})",
        f"VIOLATIONS SO FAR: {obs.get('violations_count', 0)}",
        "",
        f"ROLE INSTRUCTIONS:\n{obs.get('role_instructions', '')}",
        "",
        f"JOB DESCRIPTION:\n{obs.get('job_description', '')}",
    ]

    visible = obs.get("visible_sections") or {}
    if visible:
        lines.append("\nVISIBLE RESUME SECTIONS:")
        for sec, content in visible.items():
            lines.append(f"  [{sec.upper()}]\n{content}")

    reports = obs.get("specialist_reports") or []
    if reports:
        lines.append("\nSPECIALIST REPORTS:")
        for r in reports:
            lines.append(
                f"  [{r['specialist_role'].upper()}] has_issues={r['has_issues']} "
                f"confidence={r['confidence']:.2f}\n  {r['findings']}"
            )

    for key, label in [
        ("reference_response",    "REFERENCE CHECK RESULT"),
        ("verification_result",   "CREDENTIAL VERIFICATION"),
        ("clarification_response","CLARIFICATION"),
    ]:
        val = obs.get(key)
        if val:
            lines.append(f"\n{label}:\n{val}")

    read_details = obs.get("read_report_details") or {}
    if read_details:
        lines.append("\nFULL REPORT DETAILS:")
        for role, detail in read_details.items():
            lines.append(f"  [{role.upper()}]\n{detail}")

    feedback = obs.get("feedback")
    if feedback:
        lines.append(f"\nFEEDBACK: {feedback}")

    lines.append(f"\nAVAILABLE ACTIONS: {obs.get('available_actions', [])}")
    lines.append("\nYour JSON action:")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Rule-based agent  — used ONLY for offline data collection (no model needed)
# ─────────────────────────────────────────────────────────────────────────────
def rule_action(obs: dict) -> dict:
    """Return a reasonable rule-based action for this observation."""
    phase     = obs.get("current_phase", "")
    available = obs.get("available_actions", [])

    def pick(preferred):
        for a in preferred:
            if a in available:
                return a
        return available[0] if available else "submit_specialist_report"

    if phase == "fraud_specialist":
        choice = pick(["verify_credential", "check_reference", "submit_specialist_report"])
        if choice == "check_reference":
            return {"action_type": "check_reference", "reference_id": "ref2"}
        if choice == "submit_specialist_report":
            vr = obs.get("verification_result", "") or ""
            rr = obs.get("reference_response",  "") or ""
            bad = "FAILED" in vr or "cannot verify" in rr.lower() or "not in our system" in rr.lower()
            return {
                "action_type": "submit_specialist_report",
                "findings": f"Verification: {vr[:120]}. Reference: {rr[:120]}.",
                "has_issues": bad,
                "specialist_confidence": 0.85 if bad else 0.7,
            }
        return {"action_type": choice}

    if phase == "skills_specialist":
        viewed = set(obs.get("visible_sections", {}).keys())
        want   = next((s for s in ["experience", "education", "skills", "projects"]
                       if s not in viewed and "view_section" in available), None)
        if want:
            return {"action_type": "view_section", "section": want}
        return {
            "action_type": "submit_specialist_report",
            "findings": "Reviewed skills sections.",
            "has_issues": False,
            "specialist_confidence": 0.7,
        }

    if phase == "timeline_specialist":
        viewed = set(obs.get("visible_sections", {}).keys())
        want   = next((s for s in ["experience", "header", "summary"]
                       if s not in viewed and "view_section" in available), None)
        if want:
            return {"action_type": "view_section", "section": want}
        return {
            "action_type": "submit_specialist_report",
            "findings": "Timeline reviewed.",
            "has_issues": False,
            "specialist_confidence": 0.65,
        }

    if phase == "overseer":
        already_read = set(obs.get("reports_read", []))
        for target in ["fraud_specialist", "skills_specialist", "timeline_specialist"]:
            if target not in already_read and "read_reports" in available:
                return {"action_type": "read_reports", "report_target": target}
        reports = obs.get("specialist_reports", [])
        has_issues_count = sum(1 for r in reports if r.get("has_issues"))
        is_fraud   = has_issues_count >= 2
        return {
            "action_type": "submit_final_decision",
            "decision":   "reject" if is_fraud else "accept",
            "fraud_flag":  is_fraud,
            "confidence":  0.8,
            "fraud_reasoning": "Multiple specialists flagged issues." if is_fraud else "",
        }

    return {"action_type": available[0]} if available else {"action_type": "submit_specialist_report"}


# ─────────────────────────────────────────────────────────────────────────────
# Offline data collection  — walks episodes, stores one prompt per phase step
# ─────────────────────────────────────────────────────────────────────────────
def collect_prompts(n_episodes: int) -> list[dict]:
    """
    Run n_episodes using rule-based actions.
    Returns a list of dicts with keys: prompt, phase, is_fraud_resume, available_actions.
    """
    collected = []
    for i in range(n_episodes):
        task    = ["easy", "medium", "hard"][i % 3]
        seed    = random.randint(0, 999)
        ep_id   = f"collect-{i}-{seed}"

        try:
            resp = requests.post(
                f"{ENV_URL}/reset",
                json={"task_type": task, "seed": seed, "episode_id": ep_id},
                timeout=20,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [collect] episode {i:3d} | reset failed: {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

        obs = data.get("observation") or data   # handle flat or nested response

        steps = 0
        while steps < 20:
            phase = obs.get("current_phase", "complete")
            if phase in ("complete", None):
                break
            available = obs.get("available_actions", [])
            if not available:
                break

            # Store this observation as a training prompt
            collected.append({
                "prompt":            obs_to_prompt(obs),
                "phase":             phase,
                "available_actions": json.dumps(available),
                "steps_remaining":   obs.get("steps_remaining", 0),
            })

            # Take rule-based action
            action = rule_action(obs)
            action["episode_id"] = ep_id

            try:
                resp = requests.post(
                    f"{ENV_URL}/step",
                    json={"action": action},
                    timeout=20,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  [collect] episode {i:3d} step {steps} | step failed: {type(e).__name__}: {e}")
                break

            obs = data.get("observation") or data
            steps += 1

            if data.get("done"):
                break

        if (i + 1) % 6 == 0:
            print(f"  collected {len(collected)} prompts after {i+1} episodes …")

    return collected


# ─────────────────────────────────────────────────────────────────────────────
# Reward function  — called by GRPOTrainer for each generated completion
# Does NOT call the live environment. Scores the action quality directly.
# ─────────────────────────────────────────────────────────────────────────────
def score_completion(completion: str, phase: str, available_actions: list) -> float:
    """
    Score a single model-generated completion.
    Returns a float in [0.0, 1.0].

    Rubric:
      +0.15  Valid JSON that can be parsed
      +0.25  action_type is in available_actions (no role violation)
      +0.25  action_type is phase-optimal
      +0.20  action arguments are correct/complete
      +0.15  good reasoning text (for report/decision actions)
    """
    reward = 0.0

    # ── 1. Valid JSON ──────────────────────────────────────────────────────
    action = None
    match = re.search(r'\{[^{}]+\}', completion, re.DOTALL)
    if match:
        try:
            action = json.loads(match.group())
            reward += 0.15
        except json.JSONDecodeError:
            pass

    if action is None:
        return 0.0   # unparseable — no credit

    action_type = action.get("action_type", "")

    # ── 2. Valid action for phase (no violation) ───────────────────────────
    if action_type in available_actions:
        reward += 0.25
    else:
        return reward   # role violation — stop here

    # ── 3. Phase-optimal choice ───────────────────────────────────────────
    best = PHASE_BEST_ACTIONS.get(phase, [])
    if action_type in best[:1]:
        reward += 0.25   # first-choice optimal
    elif action_type in best:
        reward += 0.10   # acceptable

    # ── 4. Argument quality ───────────────────────────────────────────────
    if action_type == "check_reference":
        if action.get("reference_id") == "ref2":
            reward += 0.20   # ref2 is the suspicious one
        elif action.get("reference_id") == "ref1":
            reward += 0.10

    elif action_type == "view_section":
        section = action.get("section", "")
        if section in HIGH_VALUE_SECTIONS:
            reward += 0.20
        elif section:
            reward += 0.08

    elif action_type == "read_reports":
        if action.get("report_target") in ["fraud_specialist", "skills_specialist", "timeline_specialist"]:
            reward += 0.20

    elif action_type == "ask_clarification":
        q = action.get("question", "")
        if len(q) > 20:   # substantive question
            reward += 0.20
        elif q:
            reward += 0.08

    elif action_type in ("submit_specialist_report", "submit_final_decision"):
        # Must have required fields
        if action_type == "submit_specialist_report":
            has_findings  = bool(action.get("findings", "").strip())
            has_issues    = action.get("has_issues") is not None
            has_conf      = isinstance(action.get("specialist_confidence"), (int, float))
            reward += 0.10 if has_findings else 0.0
            reward += 0.05 if has_issues   else 0.0
            reward += 0.05 if has_conf     else 0.0
        else:
            has_decision  = action.get("decision") in ("accept", "reject")
            has_flag      = action.get("fraud_flag") is not None
            has_conf      = isinstance(action.get("confidence"), (int, float))
            reward += 0.10 if has_decision else 0.0
            reward += 0.05 if has_flag     else 0.0
            reward += 0.05 if has_conf     else 0.0

    # ── 5. Reasoning quality ─────────────────────────────────────────────
    reasoning = action.get("findings", "") or action.get("fraud_reasoning", "")
    fraud_keywords = {
        "failed", "denied", "fabricated", "mismatch", "unverifiable",
        "cannot verify", "not in our system", "inflated", "exaggerated",
        "conflict", "impossible", "fabrication",
    }
    if any(kw in reasoning.lower() for kw in fraud_keywords):
        reward += 0.15
    elif len(reasoning) > 30:
        reward += 0.05

    return min(reward, 1.0)


def make_reward_fn(records: list[dict]):
    """
    Build a reward function compatible with GRPOTrainer.
    GRPOTrainer calls: reward_fn(completions, prompts=..., **kwargs)
    """
    # Build lookup: prompt_text → (phase, available_actions)
    prompt_meta = {}
    for r in records:
        prompt_meta[r["prompt"]] = (
            r["phase"],
            json.loads(r["available_actions"]),
        )

    def reward_fn(completions, prompts=None, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            # prompts[i] is a list of message dicts; extract the user content
            prompt_text = ""
            if prompts and i < len(prompts):
                for msg in prompts[i]:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        prompt_text = msg.get("content", "")
                        break

            phase, available = prompt_meta.get(prompt_text, ("fraud_specialist", []))
            rewards.append(score_completion(completion, phase, available))
        return rewards

    return reward_fn


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Verify environment ────────────────────────────────────────────────
    print(f"ENV_URL  : {ENV_URL}")
    print(f"MODEL    : {MODEL_NAME}")
    try:
        health = requests.get(f"{ENV_URL}/health", timeout=15).json()
        print(f"Health   : {health}\n")
    except Exception as e:
        raise RuntimeError(f"Cannot reach environment at {ENV_URL}\n{e}")

    # ── Step 1: Collect prompts (rule-based, no model) ─────────────────────
    print(f"Collecting prompts from {N_COLLECT_EPISODES} episodes …")
    records = collect_prompts(N_COLLECT_EPISODES)
    if not records:
        raise RuntimeError(
            "No prompts collected — all episodes failed.\n"
            "Check that ENV_URL is correct and the HF Space is awake."
        )
    print(f"Collected {len(records)} prompts.\n")

    # ── Step 2: Build HF Dataset ───────────────────────────────────────────
    dataset = Dataset.from_list([
        {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": r["prompt"]},
            ]
        }
        for r in records
    ])
    print(f"Dataset  : {len(dataset)} rows")

    # ── Step 3: Load model ─────────────────────────────────────────────────
    print(f"Loading model …")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, token=HF_TOKEN or None, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        token=HF_TOKEN or None,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model loaded. Device map: {model.hf_device_map}\n")

    # ── Step 4: GRPO config ────────────────────────────────────────────────
    grpo_cfg = GRPOConfig(
        output_dir=OUTPUT_DIR,
        # Training duration
        num_train_epochs=3,
        max_steps=-1,                   # -1 = use num_train_epochs
        # Batch sizes — keep small for T4 (16 GB)
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        # GRPO-specific
        num_generations=GROUP_SIZE,     # completions per prompt
        max_completion_length=300,      # max tokens per action
        temperature=0.8,
        beta=0.04,                      # KL penalty
        # Optimiser
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=0.1,
        optim="adamw_torch",
        # Logging / saving
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        report_to="none",
        # Precision
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    )

    # ── Step 5: Reward function ────────────────────────────────────────────
    reward_fn = make_reward_fn(records)

    # ── Step 6: Train ─────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
    )

    print("Starting GRPO training …")
    train_result = trainer.train()
    print(f"\nTraining complete. Metrics: {train_result.metrics}")

    # ── Step 7: Save ──────────────────────────────────────────────────────
    save_path = f"{OUTPUT_DIR}/final"
    trainer.save_model(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")

    # ── Step 8: Plot reward curve ──────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        history = trainer.state.log_history
        steps   = [h["step"]   for h in history if "rewards/mean" in h]
        rewards = [h["rewards/mean"] for h in history if "rewards/mean" in h]

        if steps:
            os.makedirs("assets", exist_ok=True)
            plt.figure(figsize=(10, 4))
            plt.plot(steps, rewards, marker="o", markersize=3, linewidth=1.5)
            plt.xlabel("Training Step")
            plt.ylabel("Mean Reward (per step)")
            plt.title("GRPO Training — Hiring Fleet Stage 1\n(Qwen2.5-1.5B-Instruct)")
            plt.grid(True, alpha=0.4)
            plt.tight_layout()
            plt.savefig("assets/reward_curve.png", dpi=150)
            plt.show()
            print("Reward curve saved to assets/reward_curve.png")
    except Exception as e:
        print(f"Could not save plot: {e}")


if __name__ == "__main__":
    main()
