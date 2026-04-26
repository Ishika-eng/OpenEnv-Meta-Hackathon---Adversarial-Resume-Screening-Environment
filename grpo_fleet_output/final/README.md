---
language: en
license: apache-2.0
base_model: Qwen/Qwen2.5-1.5B-Instruct
tags:
  - lora
  - peft
  - grpo
  - trl
  - reinforcement-learning
  - openenv
  - resume-screening
  - multi-agent
datasets:
  - custom
pipeline_tag: text-generation
---

# Hiring Fleet — GRPO LoRA Adapter

**Base model:** `Qwen/Qwen2.5-1.5B-Instruct`  
**Method:** GRPO (Group Relative Policy Optimization) via HuggingFace TRL  
**Environment:** [IshikaMahadar/resume-env](https://huggingface.co/spaces/IshikaMahadar/resume-env)  
**GitHub:** [Ishika-eng/OpenEnv-Meta-Hackathon](https://github.com/Ishika-eng/OpenEnv-Meta-Hackathon---Adversarial-Resume-Screening-Environment)  
**Blog:** [We Made AI Detectives That Catch Fake Resumes](https://huggingface.co/spaces/IshikaMahadar/resume-env/blob/main/blog.md)

---

## What this adapter does

This LoRA adapter fine-tunes Qwen2.5-1.5B-Instruct to act as a **hiring fleet agent** — a multi-agent AI system that investigates resumes for fraud through sequential specialist roles.

The model learned through GRPO to:
- ✅ Output valid JSON actions reliably (~40% → ~95% format compliance)
- ✅ Select role-appropriate actions (reduced out-of-role violations significantly)
- ✅ Prioritise `verify_credential` as the Fraud Specialist's first move
- ✅ Write fraud indicator keywords (`failed`, `denied`, `fabricated`) in reasoning when flagging fraud

---

## The Environment

The [Hiring Fleet environment](https://huggingface.co/spaces/IshikaMahadar/resume-env) runs 4 sequential agents per episode:

| Phase | Agent | Can See | Tools |
|:---|:---|:---|:---|
| 1 | Fraud Specialist | header, references | `verify_credential`, `check_reference` |
| 2 | Skills Specialist | experience, education, skills, projects | `ask_clarification` |
| 3 | Timeline Specialist | header, summary, experience | `ask_clarification` |
| 4 | Overseer | ❌ no raw resume | `read_reports`, `submit_final_decision` |

The Overseer can't see the resume — it must reason purely from specialist reports. If the specialists write poor reports, the Overseer has no signal. The chain of reasoning is real.

---

## Training Details

| Parameter | Value |
|:---|:---|
| Base model | Qwen/Qwen2.5-1.5B-Instruct |
| Adapter type | LoRA (r=16, alpha=32) |
| Target modules | q_proj, v_proj |
| Trainable parameters | 2,179,072 (0.14% of total) |
| Framework | HuggingFace TRL — GRPOTrainer |
| Hardware | T4 GPU (Google Colab free tier) |
| Training steps | 792 |
| Epochs | 2 |
| Group size | 4 completions per prompt |

**Reward curve:**

![GRPO Reward Curve](https://github.com/Ishika-eng/OpenEnv-Meta-Hackathon---Adversarial-Resume-Screening-Environment/raw/main/assets/reward_curve.png)

| Metric | Value |
|:---|:---|
| Start reward | 0.736 |
| Best reward | 0.850 |
| Improvement | +15.5% |

---

## Evaluation

Evaluated against the live HF Space environment — 9 episodes (3 per difficulty tier):

| Agent | Easy | Medium | Hard | Overall |
|:---|:---:|:---:|:---:|:---:|
| Rule-based baseline | 0.747 | 0.873 | 1.000 | **0.873** |
| Fine-tuned (this adapter) | 0.722 | 0.888 | 1.000 | **0.870** |

The trained model matches the hand-coded expert baseline despite learning purely from rewards — no hard-coded logic. On medium difficulty it outperforms the baseline (0.888 vs 0.873).

---

## How to use

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = "Qwen/Qwen2.5-1.5B-Instruct"
adapter    = "IshikaMahadar/hiring-fleet-grpo-adapter"

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model     = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
model     = PeftModel.from_pretrained(model, adapter)
model.eval()

# The model expects a JSON action as output given an observation prompt
# See inference_fleet.py in the GitHub repo for full multi-agent inference
```

See [`inference_fleet.py`](https://github.com/Ishika-eng/OpenEnv-Meta-Hackathon---Adversarial-Resume-Screening-Environment/blob/main/inference_fleet.py) for complete inference code with any OpenAI-compatible model API.

---

## Training notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://huggingface.co/spaces/IshikaMahadar/resume-env/blob/main/train_grpo_fleet.ipynb)

Run on Colab free tier (T4 GPU). ~2 hours to complete.

---

*Built at OpenEnv Meta Hackathon 2026 — Team SmartBytes (Ishika Mahadar · Prisha Parikh · Saee Kolhapure)*
