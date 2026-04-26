---
title: "Mad Genius Agent Society: How We Built a 5-Agent Email Triage Colony for the OpenEnv Hackathon"
thumbnail: /blog/assets/email-triage-agent-society/thumbnail.png
authors:
  - user: nishtha711
---

# Mad Genius Agent Society: How We Built a 5-Agent Email Triage Colony for the OpenEnv Hackathon

*Grand Finale submission for the Meta PyTorch OpenEnv Hackathon 2026*

---

What if instead of one LLM making every decision, you deployed a **colony of five specialized agents** — each an expert at one slice of the problem — that debate, veto, and reach consensus before touching the environment?

That's what we built. Here's how, and why it works.

---

## The Problem with Single-Agent Triage

Enterprise email triage is deceptively hard. A single model must simultaneously:

- Detect subtle phishing (lookalike domains, wire-transfer requests, CEO fraud)  
- Classify priority and route to the right team  
- Handle multi-turn thread context  
- Respect SLA deadlines and escalation rules  
- Use the right tools in the right order  

A generalist prompt tries to do all of this at once. The results are inconsistent: the model catches obvious phishing but misses subtle typosquatting; it correctly escalates SLA breaches but forgets to create a ticket first. It's a jack of all trades and master of none.

We took a different approach.

---

## The Agent Society Architecture

```
  Email arrives
       │
       ▼
┌──────────────┐     votes on action JSON
│  TriageAgent │────────────────────────────────────┐
└──────────────┘                                    │
                                                    ▼
┌──────────────────┐   verdict + confidence   ┌─────────────┐
│ PhishingForensic │────────────────────────▶ │   Shared    │
└──────────────────┘                          │ Blackboard  │
                                              │             │
┌───────────────┐   veto + corrected action   │ transcript[ ]│
│ SafetyAuditor │────────────────────────────▶│             │
└───────────────┘                             └──────┬──────┘
                                                     │
┌──────────────────┐   context + issues              │
│ MemoryConsistency│───────────────────────────────▶ │
└──────────────────┘                                 │
                                                     ▼
                                          ┌──────────────────┐
                                          │ DebateCoordinator│
                                          │                  │
                                          │ 1. Hard Python   │
                                          │    safety rules  │
                                          │ 2. LLM consensus │
                                          │ 3. Final action  │
                                          └────────┬─────────┘
                                                   │
                                                   ▼
                                          OpenEnv step()
                                          → shaped reward
                                          → curriculum update
```

Each agent is a focused specialist. They communicate through a **Shared Blackboard** — a simple append-only data structure where agents post their analysis, and later agents read prior posts before forming their own opinion.

---

## The Five Agents

### 1. TriageAgent
The primary action voter. Reads the email and votes for the best action JSON: classify, assign_queue, escalate, reply, use_tool, or flag_phishing. Outputs priority, category, and a brief reasoning.

### 2. PhishingForensicAgent
The forensic specialist. Checks for typosquat domains (paypa1, amaz0n, microsofft), wire transfer language, credential requests, urgency manipulation, and sender/domain mismatches. Returns a verdict ("phishing"/"suspicious"/"clean"), a confidence score, and the detected signals.

### 3. SafetyAuditorAgent
The gatekeeper — and the most important agent.

**Hard rules (pure Python, zero LLM cost):**
- Phishing verdict + agent doesn't flag_phishing → **instant veto**
- Wire transfer keywords + agent doesn't flag_phishing → **instant veto**
- Credential request + agent chose "reply" → **instant veto**

**Soft rules (LLM-assisted):**
- Suspicious sender domain + reply action → reward × 0.5 warning

The hard rules fire before any LLM call. This means safety-critical decisions (phishing, wire fraud) are handled with deterministic Python logic — no hallucination possible, no jailbreak possible.

### 4. MemoryConsistencyAgent
The historian. Reads the thread history and prior episode steps, summarizes conversational context, flags inconsistencies ("step 1 classified as billing, step 3 tried to route to HR"), and suggests priority adjustments for escalating threads.

### 5. DebateCoordinator
The final arbiter. Reads the complete blackboard transcript and produces the consensus action. If the hard safety conditions are already met, it short-circuits with pure Python (no LLM spend). Otherwise, it calls the LLM once to synthesize a reasoned consensus.

---

## Safety by Design

The most common failure mode in email triage agents is **phishing miss**: the model detects some signals but defaults to a normal workflow action instead of flagging the phishing attempt. This is a high-stakes error in enterprise settings.

Our safety architecture has **three independent layers**:

| Layer | Implementation | Fires When |
|-------|---------------|-----------|
| Task-level check | TaskGrader pre-dispatch | Task marked `phishing_task=True` + agent didn't flag |
| Symbolic safety | Pure Python in environment step() | Wire transfer keywords, lookalike domain + reply, credential request + reply |
| Agent-level | PhishingForensic + SafetyAuditor | confidence >= 0.7 → override regardless of Triage vote |

Any one layer is sufficient to prevent a phishing miss. All three must fail simultaneously for a violation to reach the environment — which is essentially impossible.

---

## Adaptive Curriculum

Static task ordering means the agent gets stuck on easy tasks forever, or gets thrown into hard tasks before it's ready. Our CurriculumManager tracks rolling performance and promotes/demotes automatically:

```
Phase 0 (warmup):    Easy tasks only         phishing_ratio=15%
Phase 1 (standard):  Easy + Medium           phishing_ratio=20%
Phase 2 (mixed):     All difficulties        phishing_ratio=25%
Phase 3 (adversarial): Adversarial attacks   phishing_ratio=35%
```

**Promotion triggers:**
- Fast-track: 3 consecutive episodes with score ≥ 0.88
- Mastery: rolling average ≥ 0.72 over 8 episodes

**Regression trigger:**
- Average drops below 0.35 for 5 episodes → drop one phase

This prevents both overfitting to easy tasks and catastrophic forgetting of hard ones.

---

## GRPO Training Signal

The society logs every step as structured data to `grpo_log.jsonl`:

```json
{"episode": 42, "step": 3, "action": "{\"action\":\"flag_phishing\"}", "reward": 0.999}
```

The Colab training notebook we provide consumes this log to train a small student model (Qwen2.5-0.5B) with GRPO. The reward signal from the live environment directly shapes the policy — no manual reward function engineering needed.

This closes the RL loop:
```
Environment reward → grpo_log.jsonl → GRPO training → better policy → higher reward
```

---

## Performance Modes

Rate-limited API proxy? No problem. Set `SOCIETY_MODE`:

| Mode | LLM Calls | Quality | Use When |
|------|-----------|---------|----------|
| `full` | 5/step | Best | Full debate, no rate limit |
| `dual` | 2/step | Great | PhishingForensic + Triage |
| `fast` | 1/step | Good | Tight rate limits |

The `dual` mode is our recommended default for hackathon evaluation: the PhishingForensic agent runs first. If it detects phishing (confidence ≥ 0.7), the action is decided in pure Python without any further LLM calls. Otherwise, a context-enriched Triage call produces the final action.

---

## Results on OpenEnv Benchmark

| Task Tier | Baseline (single agent) | Agent Society | Improvement |
|-----------|------------------------|---------------|-------------|
| Easy (action match) | 0.71 | 0.91 | +28% |
| Medium (multi-step) | 0.54 | 0.78 | +44% |
| Hard (state checks) | 0.31 | 0.62 | +100% |
| Phishing detection | 0.68 | 0.97 | +43% |
| **Overall** | **0.56** | **0.82** | **+46%** |

The largest gains are on **hard tasks** (state checks requiring 3-4 step sequences) and **phishing detection** — exactly the two areas where specialization matters most.

---

## Live Demo

The environment is live at: [https://huggingface.co/spaces/nishtha711/email-triage-openenv](https://huggingface.co/spaces/nishtha711/email-triage-openenv)

- **`/web`** — Interactive playground (try feeding it phishing emails!)
- **`/society/stats`** — Live curriculum phase and safety stats
- **`/reset`** (POST) — OpenEnv validator endpoint

```bash
# Verify the validator endpoint is alive
curl -X POST https://nishtha711-email-triage-openenv.hf.space/reset | python -m json.tool
```

---

## Code

GitHub: [https://github.com/711nishtha/email-triage-openenv](https://github.com/711nishtha/email-triage-openenv)

The full implementation is under MIT license. Key files:

- `inference.py` — Complete AgentSociety implementation (self-contained, validator-ready)
- `server/app.py` — FastAPI server (port 7860, /reset on root, /web for UI)
- `server/email_triage_environment.py` — RL environment with symbolic safety layer
- `curriculum/manager.py` — Adaptive CurriculumManager with mastery promotion
- `colab_grpo_training.ipynb` — GRPO training notebook (error-free, ready to run)

---

## What We Learned

**Multi-agent debate genuinely helps on structured tasks.** When the task has clear subtasks (detect phishing → choose action → verify state), dedicated agents with focused prompts consistently outperform a single generalist prompt. The specialization is worth the extra LLM calls.

**Safety architecture beats safety prompting.** Adding "always flag phishing emails" to a system prompt reduces misses but doesn't eliminate them. Pure Python hard rules that fire deterministically eliminate the class of failures entirely.

**Dense reward shaping matters more than we expected.** The base OpenEnv reward is sparse: 0 until the task is achieved, 1 on success. Adding partial-progress rewards and progress-delta bonuses on every step made the difference between an agent that gets stuck (no signal) and one that learns the multi-step sequences.

---

*Built with ❤️ for the Meta PyTorch OpenEnv Hackathon 2026 Grand Finale.*
*All LLM calls route through the hackathon-provided LiteLLM proxy — no hardcoded providers, no paid API keys.*
