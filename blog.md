# We Made AI Detectives That Catch Fake Resumes 🕵️

*by Team SmartBytes — Ishika Mahadar, Prisha Parikh, Saee Kolhapure*

---

Okay so here's the thing that got us thinking.

Companies use AI to screen resumes these days. Makes sense right? Thousands of applications, can't read them all manually. But here's the problem — people have figured out how to fool these AI screeners. And it's actually not that hard.

The trick is, you don't put all your lies in one place.

You put your fake university name in the header. Looks fine, just a name.

You put dates that overlap between two jobs in the experience section. Easy to miss.

You put a reference number who's going to lie for you in the references section. Nobody cross-checks.

And the AI reads the whole thing top to bottom, goes "yeah looks good!" and passes the candidate through.

**No single section looks wrong. The fraud only shows up when you connect the dots between ALL the sections.**

And that's the exact thing a single AI agent just doesn't do.

---

## So we thought — what if we built a whole investigation team instead?

Like, what if instead of one AI reading the whole resume, we had different specialist AIs each responsible for one specific part? Each one focuses on their area, does their investigation, writes up their findings — and then a boss AI reads all the reports and makes the final call.

That's exactly what we built. We call it **Hiring Fleet**.

It runs 4 agents, one after another, on every resume. Here's how the sequence works:

---

### Agent 1 — The Fraud Specialist 🔍

This agent goes first. It can ONLY see the header and the references section. That's it. It doesn't get to read the person's experience or skills — that's not its job.

What it can do:
- Call `verify_credential` — checks if the university is real, if the employer actually exists, if certifications are legit
- Call `check_reference ref1` or `check_reference ref2` — literally calls the references and asks about the candidate
- View the header and references sections

It has a limited number of moves (2–3 depending on difficulty). So it has to be smart about which actions to take first.

When it's done, it writes a **specialist report** — "I found this, I think there are issues, here's my confidence level."

---

### Agent 2 — The Skills Specialist 💡

Now this agent comes in. It can ONLY see the experience, education, skills, and projects sections. It never sees what the Fraud Specialist found.

What it can do:
- View any of those four sections
- Ask a clarification question to the candidate (like "explain this gap" or "what did you build here")

Its job is to figure out — does this person actually have the skills the job needs? Does their experience make sense for the role?

It also writes a specialist report at the end.

---

### Agent 3 — The Timeline Specialist 📅

Third in line. It looks at the header, summary, and experience sections specifically hunting for timeline problems.

Was someone claiming to work two full-time jobs at the same time? Is there a 2-year unexplained gap? Does their career progression make zero sense?

It can ask clarification questions too — "you say you were at Amazon 2018–2023 but also Google 2020–2022, how does that work?"

Writes its report. Passes to the boss.

---

### Agent 4 — The Overseer ⚖️

Here's the most interesting part. The Overseer **cannot see the resume at all.**

We mean that literally. It has no access to any resume section. Zero.

All it can do is:
- `read_reports` — read what each specialist wrote
- `request_reinvestigation` — send one specialist back for a second look (once per episode)
- `submit_final_decision` — hire or reject, with a fraud flag and reasoning

So if the specialists wrote garbage reports, the Overseer has nothing to work with. The whole chain has to work properly for the final decision to be right.

This is what makes it genuinely hard and genuinely interesting.

---

## Okay but how does the AI know if it did a good job?

This is where the reward system comes in. Every action earns or loses points, and the total at the end tells us how well the agent team did.

Some examples of what earns points:
- `verify_credential` comes back with a FAILED result → **+0.05** (you found evidence!)
- `check_reference` reveals the reference is denying the employment → **+0.05** (smoking gun!)
- Overseer reads all 3 specialist reports before deciding → bonus points
- Final decision is correct (right hire/reject + right fraud flag) → **+0.35 + 0.25**
- All 4 agents get it right simultaneously → **fleet coordination bonus +0.08**

And what loses points:
- Trying an action that's not allowed in your current phase → **−0.05** (called a violation)
- Submitting a fraud flag but writing zero reasoning → penalty
- Being super confident but being wrong → penalty

Total possible range is 0.0 to 1.0. No AI judge scoring this — it's all calculated from ground truth data. Completely reproducible.

---

## The dataset — what resumes are we actually testing on?

We made 36 resumes across 3 difficulty levels. 12 per level. About 42% of them are fraud (so the model can't just always say "fraud" and win).

**Easy** — obvious stuff. Fake university that doesn't exist. Reference who says "I've never heard of this person." Dates that literally don't add up.

**Medium** — more subtle. Real company, but the title is exaggerated. Skills that are half-true. A reference that gives vague non-answers.

**Hard** — genuinely sophisticated. Someone claims Director but reference says they were a Senior Engineer. Multiple sections that are each individually plausible but together tell a different story.

Every fraud resume is guaranteed to have at least one detectable signal — so the Fraud Specialist always has something to find if it takes the right actions.

---

## The training part — we actually taught a model to play this

We took **Qwen 2.5 — 1.5 billion parameters** — and trained it on this environment using something called GRPO.

GRPO basically works like this: give the model a situation, generate a bunch of different possible actions it could take, score each one using the reward system, and then update the model to make the high-scoring actions more likely in the future. Repeat thousands of times.

We ran this on a **free T4 GPU on Google Colab**. Took about 2 hours. 984 training steps.

Here's what the reward looked like over training:

![GRPO Reward Curve — reward going up from 0.736 to 0.850 over 984 steps](https://github.com/Ishika-eng/OpenEnv-Meta-Hackathon---Adversarial-Resume-Screening-Environment/raw/main/assets/reward_curve.png)

Started at 0.736. Got to 0.850. That's a **+15.5% improvement**.

---

## What actually changed after training?

Before training the model was basically doing random stuff — outputting malformed JSON, picking actions that weren't even available, trying to do skills work during the fraud phase.

After training:

✅ It outputs valid JSON actions almost every single time (went from ~40% to ~95%)

✅ As Fraud Specialist, its first move is almost always `verify_credential` — because that's the highest-signal action available. Check if credentials are real before anything else.

✅ It stopped making role violations — it learned that as Skills Specialist you don't call references, that's not your job

✅ When it flags fraud, it actually writes out *why* — using keywords like "failed", "denied", "fabricated"

---

## How does it compare to our baseline?

We built a rule-based agent (basically hand-coded logic — "always verify first, always check ref2 on fraud cases") as a baseline. Then we compared:

![Before vs After comparison chart](https://github.com/Ishika-eng/OpenEnv-Meta-Hackathon---Adversarial-Resume-Screening-Environment/raw/main/assets/comparison_chart.png)

| Agent | Easy | Medium | Hard | Overall |
|---|---|---|---|---|
| Rule-based (hand-coded) | 0.747 | 0.873 | 1.000 | **0.873** |
| Our trained model (GRPO) | 0.722 | 0.888 | 1.000 | **0.870** |

The trained model basically **matches the hand-coded expert system** — but it learned this from rewards alone. No hard-coded logic. No one told it "always verify first." It figured that out by itself through training.

On medium difficulty it actually beats the baseline (0.888 vs 0.873). The model learned to be smarter about which step to take when, rather than always following the same fixed sequence.

---

## A real episode, step by step

Here's what actually happens when you run a **medium-difficulty fraud resume** through the fleet. This candidate claims to be a Senior Machine Learning Engineer at a real tech company with a Stanford degree — but it's fabricated.

**Reset:** `POST /reset  {"task_type": "medium", "seed": 10}`

---

**Phase 1 — Fraud Specialist** *(2 steps)*

> *Step 1:* `verify_credential`
> → **`"FAILED — institution not found in accredited registry"`**
> → Reward: **+0.05** (fraud signal found!)

> *Step 2:* `submit_specialist_report`
> → findings: `"Credential check: FAILED — institution not found. Suspicious."`, has_issues: `true`, confidence: `0.88`
> → Reward: **+0.08** (correct fraud detection, calibrated confidence)

The Fraud Specialist used both its steps efficiently: verified first, submitted immediately when it found a failure. No wasted moves.

---

**Phase 2 — Skills Specialist** *(3 steps)*

> *Step 1:* `view_section  experience`
> → Sees: 4 years at company, lists "led ML pipeline serving 10M users"

> *Step 2:* `view_section  skills`
> → Sees: Python, TensorFlow, Kubernetes — all plausible

> *Step 3:* `submit_specialist_report`
> → findings: `"Skills match job requirements. Experience description is plausible."`, has_issues: `false`, confidence: `0.70`
> → Reward: **+0.03**

The Skills Specialist sees nothing obviously wrong — the fraud is well-hidden in the credentials, not the skills section. This is exactly the design: fraud hides across sections.

---

**Phase 3 — Timeline Specialist** *(3 steps)*

> *Step 1:* `view_section  experience`
> → Sees dates: Jan 2019 – Dec 2023

> *Step 2:* `view_section  header`
> → Sees: graduated 2018, consistent with employment start

> *Step 3:* `submit_specialist_report`
> → findings: `"Timeline is consistent. No overlapping dates."`, has_issues: `false`, confidence: `0.65`
> → Reward: **+0.01**

No timeline issues — again, the fraud is in the credential, not the dates.

---

**Phase 4 — Overseer** *(3 steps)*

> *Step 1:* `read_reports  fraud_specialist`
> → Receives: `"FAILED credential. has_issues: true. confidence: 0.88"`
> → Reward: **+0.02**

> *Step 2:* `read_reports  skills_specialist` + `read_reports  timeline_specialist`
> → Both clean. Only Fraud Specialist flagged an issue.
> → Reward: **+0.05** (read all 3 reports)

> *Step 3:* `submit_final_decision`
> → decision: `"reject"`, fraud_flag: `true`, confidence: `0.85`
> → fraud_reasoning: `"Fraud flagged by fraud_specialist: FAILED credential check"`
> → Reward: **+0.60** (correct reject + correct fraud flag + calibrated confidence)

---

**Episode total reward: ~0.84 / 1.0** ✅

The key insight: the Fraud Specialist found the lie in Step 1. The Overseer correctly synthesised that one signal from three reports — even though the other two specialists found nothing wrong — and made the right call. The chain worked.

---

## Where to find everything

🌐 **Live Environment (try it right now):**
https://huggingface.co/spaces/IshikaMahadar/resume-env

🤖 **Trained LoRA Adapter:**
https://huggingface.co/IshikaMahadar/hiring-fleet-grpo-adapter

💻 **Full Code:**
https://github.com/Ishika-eng/OpenEnv-Meta-Hackathon---Adversarial-Resume-Screening-Environment

📓 **Training Notebook (Colab):**
https://github.com/Ishika-eng/OpenEnv-Meta-Hackathon---Adversarial-Resume-Screening-Environment/blob/main/train_grpo_fleet.ipynb

---

## One last thing — why does this actually matter?

Every big company has a version of this process already. Background check team. Technical screener. HR coordinator. Hiring manager who makes the final call.

It's just humans doing it. It's slow, expensive, and inconsistent.

What we built is the training environment for teaching AI to do this work — properly, with role boundaries, with a reward system that punishes shortcuts, and with an oversight structure that requires the whole chain to work, not just one agent.

The Overseer can't cheat by peeking at the resume. The Fraud Specialist can't steal the Skills Specialist's sections. Everyone has to do their own job well.

That's kind of the whole point. 🙂

---

*Built at OpenEnv Meta Hackathon 2026 — Team SmartBytes*
*Ishika Mahadar · Prisha Parikh · Saee Kolhapure*