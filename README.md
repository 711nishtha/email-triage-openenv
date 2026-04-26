---
title: Enterprise Email Triage RL Environment
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
---

# Email Triage Agent Society
### Meta PyTorch OpenEnv Hackathon 2026 — Grand Finale Submission

[![HF Space](https://img.shields.io/badge/🤗%20Space-Live-blue)](https://huggingface.co/spaces/nishtha711/email-triage-openenv)
[![Blog Post](https://img.shields.io/badge/📝%20Blog-HuggingFace-orange)](https://huggingface.co/blog/nishtha711/email-triage-agent-society)
[![GitHub](https://img.shields.io/badge/GitHub-711nishtha-black)](https://github.com/711nishtha/email-triage-openenv)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## Submission Materials

| Resource | Link |
|----------|------|
| 🤗 **Live HuggingFace Space** | https://huggingface.co/spaces/nishtha711/email-triage-openenv |
| 📝 **HuggingFace Blog** | [https://huggingface.co/blog/nishtha711/email-triage-agent-society](https://huggingface.co/spaces/nishtha711/email-triage-openenv/blob/main/BLOG.md) |
| 💻 **GitHub Repo** | https://github.com/711nishtha/email-triage-openenv |
| 📓 **Colab Training Notebook** | `colab_grpo_training.ipynb` (in this repo — run to reproduce all plots) |

---

## Why Enterprise Email Triage?

Enterprise support teams process thousands of emails daily. Misrouted billing queries sit for days.
Phishing attacks get replied to. SLA breaches go unescalated. A well-trained RL agent that correctly
classifies, routes, flags phishing, escalates, and sequences tool calls can save enormous cost and
prevent security incidents.

This environment makes it hard in the right ways:
- **30 tasks** (10 easy / 10 medium / 10 hard) with machine-readable grading
- **Phishing tasks** require the agent to resist normal workflows and flag instead
- **Multi-step tasks** require correct tool sequencing (CRM lookup → assign queue)
- **Hard state-check tasks** verify that the correct artifacts were actually created
- **Adaptive curriculum** promotes the agent from easy → adversarial as mastery grows

---

## The Agent Society

Instead of one LLM making every decision, we deploy **5 specialized agents** on a Shared Blackboard:

```
  Email arrives
       │
  TriageAgent ─────────────────────────────┐
  PhishingForensicAgent ───────────────────┤
  SafetyAuditorAgent ──────────────────────┤──► Shared Blackboard ──► DebateCoordinator ──► action
  MemoryConsistencyAgent ──────────────────┘         (transcript)          │
                                                                     ① Python hard rules
                                                                     ② LLM consensus
                                                                     ③ Verified action JSON
```

| Agent | Specialisation | LLM calls/step |
|-------|---------------|---------------|
| TriageAgent | Action vote + priority + category | 1 |
| PhishingForensicAgent | Typosquat/wire/credential/urgency detection | 1 |
| SafetyAuditorAgent | Hard Python veto + LLM soft veto | 0–1 |
| MemoryConsistencyAgent | Thread history, contradiction detection | 1 |
| DebateCoordinator | Consensus synthesis + final action | 0–1 |

**Performance modes** (`SOCIETY_MODE` env var):

| Mode | LLM calls | Use when |
|------|-----------|----------|
| `full` | 5 | Full 5-agent debate |
| `dual` | 2 | Best quality/latency tradeoff |
| `fast` | 1 | Tight rate limits |

---

## Safety by Design

Three independent layers — ALL must fail simultaneously for a phishing miss:

1. **Task-level** (environment): `phishing_task=True` + wrong action → REWARD_MIN immediately
2. **Symbolic** (pure Python): wire transfer / credential / lookalike domain → instant Python veto
3. **Agent-level** (LLM): PhishingForensic confidence ≥ 0.7 → DebateCoordinator Python override

No hallucination possible on safety-critical decisions. No jailbreak can bypass layer 2.

---

## Adaptive Curriculum

```
Phase 0 (warmup):      Easy only         phishing_ratio=15%
Phase 1 (standard):    Easy + Medium     phishing_ratio=20%
Phase 2 (mixed):       All difficulties  phishing_ratio=25%
Phase 3 (adversarial): Adversarial       phishing_ratio=35%
```

Promotion: 3× score ≥ 0.88 (fast-track) OR rolling avg ≥ 0.72 over 8 episodes
Regression: avg < 0.35 for 5 episodes → drop one phase

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start episode — **OpenEnv validator (root level)** |
| `/step` | POST | Send action, get reward |
| `/state` | GET | Current environment state |
| `/web` | GET | Interactive playground UI |
| `/society/stats` | GET | Live curriculum + safety stats (JSON) |
| `/society/log` | GET | Safety violation audit log (JSON) |
| `/health` | GET | Liveness check |

```bash
# Verify validator endpoint is alive
curl -X POST https://nishtha711-email-triage-openenv.hf.space/reset | python -m json.tool
```

---

## Quick Start

```bash
pip install -r requirements.txt

# Local testing with Groq 
export API_BASE_URL=https://api.groq.com/openai/v1
export API_KEY=your_groq_key
export MODEL_NAME=llama-3.1-8b-instant
export SOCIETY_MODE=dual

uvicorn server.app:app --host 0.0.0.0 --port 7860
python inference.py
```

---

## Results

| Task Tier | Single Agent | Agent Society | Δ |
|-----------|-------------|---------------|---|
| Easy | 0.71 | 0.91 | +28% |
| Medium | 0.54 | 0.78 | +44% |
| Hard | 0.31 | 0.62 | +100% |
| Phishing | 0.68 | 0.97 | +43% |
| **Overall** | **0.56** | **0.82** | **+46%** |

---

## Structure

```
email-triage-openenv/
├── inference.py                     # Agent Society (validator entry point)
├── models.py                        # Pydantic data models
├── client.py                        # OpenEnv client
├── server/
│   ├── app.py                       # FastAPI (port 7860, /reset on root, /web for UI)
│   ├── email_triage_environment.py  # RL environment with safety layer
│   └── services/
│       ├── tasks.py                 # 30 task definitions
│       ├── task_grader.py           # Dense reward engine
│       ├── episode_tracker.py       # Per-episode tracker
│       └── curriculum.py           # Legacy curriculum fallback
├── agents/society.py                # AgentSociety re-export
├── curriculum/manager.py            # Adaptive CurriculumManager
├── colab_grpo_training.ipynb        # GRPO training notebook
├── BLOG_POST.md                     # HuggingFace blog post source
├── requirements.txt
├── openenv.yaml
└── Dockerfile
```

