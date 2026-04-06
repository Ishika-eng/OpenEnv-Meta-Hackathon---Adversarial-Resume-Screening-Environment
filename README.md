---
title: Adversarial Resume Screening Environment
emoji: "\U0001F4C4"
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
tags: [openenv]
---

# Adversarial Resume Screening

## Overview & Motivation

Automated hiring systems are increasingly targeted by **adversarial resumes** — CVs crafted with fabricated credentials, inflated titles, and keyword stuffing to bypass AI filters. This environment benchmarks how well AI agents can **investigate, verify, and make robust hiring decisions** under adversarial conditions.

Unlike single-shot classifiers, agents must conduct **multi-step investigations**: reviewing resume sections, checking references, verifying credentials, and asking clarifying questions before reaching a decision. This mirrors the real-world HR due diligence process.

---

## Environment Design

### Multi-Step Episode Flow

Each episode follows an investigation workflow:

1. **Initial**: Agent receives the job description and candidate header
2. **Investigation**: Agent gathers evidence through 4 action types
3. **Decision**: Agent submits final accept/reject with fraud assessment

Episodes have a **step budget** that varies by difficulty (easy: 6, medium: 8, hard: 10). If the budget runs out without a decision, the episode ends with zero reward.

### Observation Space (`ResumeObservation`)

| Field | Type | Description |
|:---|:---|:---|
| `task_type` | `"easy"\|"medium"\|"hard"` | Difficulty tier |
| `phase` | `"initial"\|"investigation"\|"decision_made"` | Episode phase |
| `job_description` | `string` | Role requirements |
| `visible_sections` | `Dict[str, str]` | Resume sections revealed so far |
| `available_actions` | `List[str]` | Valid action types |
| `clarification_response` | `string\|null` | Answer to last clarification |
| `reference_response` | `string\|null` | Last reference check result |
| `verification_result` | `string\|null` | Last credential verification |
| `steps_remaining` | `int` | Step budget countdown |
| `feedback` | `string\|null` | Environment hints/warnings |

### Action Space (`ResumeAction`)

| Action Type | Fields | Effect |
|:---|:---|:---|
| `view_section` | `section` (header/summary/experience/education/skills/projects/references) | Reveals a resume section |
| `ask_clarification` | `question` (free text) | Returns candidate's answer |
| `check_reference` | `reference_id` (ref1/ref2) | Returns reference response |
| `verify_credential` | — | Returns education/employment/cert verification status |
| `submit_decision` | `decision`, `fraud_flag`, `confidence`, `fraud_reasoning` | Terminal action, ends episode |

### Reward Function

Rewards accumulate across the episode from two sources:

**Investigation rewards (per-step):**
| Action | Reward |
|:---|:---|
| View high-value section (experience/education/skills) | +0.03 |
| View other section | +0.01 |
| Relevant clarification answer | +0.03 |
| Generic clarification answer | +0.01 |
| Reference check (fraud case) | +0.05 |
| Reference check (non-fraud) | +0.02 |
| Credential verification (reveals failure) | +0.05 |
| Credential verification (all pass) | +0.02 |

**Terminal decision reward:**
| Component | Reward |
|:---|:---|
| Decision correct | +0.35 |
| Decision wrong | -0.35 |
| Fraud flag correct | +0.25 |
| Fraud flag wrong | -0.25 |
| Confidence calibration (>= 0.7 + both correct) | +0.10 |
| Investigation thoroughness (3+ sections + ref/verify) | +0.10 |
| Fraud reasoning quality (mentions indicators) | +0.10 |
| Early termination penalty (submit on step 1) | -0.15 |

**Total reward range**: [-1.0, 1.0] per episode.

---

## Task Difficulty & Graders

| Tier | Resumes | Step Budget | Description |
|:---|:---:|:---:|:---|
| **Easy** | 12 | 6 | Clear match/mismatch, obvious fraud (impossible timelines, fake institutions) |
| **Medium** | 12 | 8 | Subtle skill gaps, partial matches, embellished but plausible resumes |
| **Hard** | 12 | 10 | Title inflation, scope exaggeration, references that contradict claims, sophisticated fabrication |

Graders are **deterministic**: all reward computation uses ground-truth fields in the dataset (expected_decision, is_fraud, fraud_indicators). No LLM judge is needed.

---

## Setup & Usage

### Local Setup

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Docker Setup

```bash
docker build -t resume-env .
docker run -p 7860:7860 resume-env
```

### Running Inference

Configure environment variables:
```bash
export API_BASE_URL="https://api.groq.com/openai/v1"   # or your LLM endpoint
export MODEL_NAME="llama-3.3-70b-versatile"             # or your model
export HF_TOKEN="your-api-key"
export ENV_URL="http://localhost:7860"                   # or your HF Space URL
```

Run the baseline agent:
```bash
python inference.py
```

The script runs 9 episodes (3 per difficulty tier) and emits structured `[START]`/`[STEP]`/`[END]` logs.

---

## Baseline Scores

Evaluated using **Llama-3.3-70B** via Groq across 9 episodes (3 easy, 3 medium, 3 hard):

| Tier | Avg Score | Avg Steps |
|:---|:---:|:---:|
| Easy | ~0.80 | 4-5 |
| Medium | ~0.65 | 5-6 |
| Hard | ~0.45 | 6-8 |
| **Overall** | **~0.63** | ~5.5 |

Hard-tier resumes with subtle title inflation and reference contradictions consistently challenge frontier models.

---

## API Endpoints

| Method | Path | Description |
|:---|:---|:---|
| POST | `/reset` | Reset environment, returns initial observation |
| POST | `/step` | Submit action, returns observation with reward |
| GET | `/state` | Get current internal state |
| GET | `/health` | Health check |
| GET | `/` | Web UI |

---

**OpenEnv Compliance**: v2.0.0
**Deployment**: [ishikamahadar-resume-env.hf.space](https://huggingface.co/spaces/IshikaMahadar/resume-env)
