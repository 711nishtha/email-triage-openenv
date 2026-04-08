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

# Advanced Enterprise Email Triage RL Environment

An **OpenEnv** reinforcement learning environment where AI agents learn to triage enterprise emails with real-world complexity: thread awareness, phishing detection, tool usage, priority routing, and SLA management across **3 difficulty tiers (30+ tasks)**.

> **[API Docs](http://localhost:7860/docs)** | **[Playground](http://localhost:7860/web)**

---

## Overview

The agent receives email observations and must decide how to handle them — classify priority, detect phishing, assign to queues, trigger tools (calendar, CRM, escalation), or compose replies. The environment rewards correct triage, penalizes misclassification, and enforces phishing safety guardrails.

Each **action** is a JSON string with an `action` field, e.g.:

```json
{"action": "classify", "priority": "high", "category": "customer_complaint"}
{"action": "flag_phishing"}
{"action": "assign_queue", "queue": "billing"}
{"action": "escalate", "to": "manager", "reason": "SLA breach risk"}
{"action": "use_tool", "tool": "calendar", "params": {"search": "meeting"}}
{"action": "reply", "tone": "professional", "summary": "acknowledged"}
```

---

## Task Tiers (30 Tasks)

### Easy — 10 tasks

> Single-step classification and routing

- Identify the correct priority (low / medium / high / urgent)
- Classify category (billing, support, sales, hr, it, general)
- Detect obvious phishing emails (suspicious links, impersonation)
- Route to the correct queue
- Graded by **action_match** — checks correct action + parameter

### Medium — 10 tasks

> Multi-step workflows with tool usage

- Thread-aware triage: interpret follow-up emails in context of prior conversation
- Trigger the right tool (calendar lookup, CRM query, ticket creation)
- Compose an appropriately-toned reply before routing
- Graded by **multi_step** — validates ordered sequence of actions

### Hard — 10 tasks

> Complex reasoning: phishing, SLA breaches, and ambiguous escalations

- Detect subtle phishing (lookalike domains, urgency manipulation)
- Manage SLA deadlines: identify breach risk and escalate correctly
- Handle ambiguous emails requiring clarification requests
- Multi-tool pipelines: CRM lookup → ticket → notify
- Graded by **state_checks** — verifies end-state flags and tool call log

---

## Reward Shaping

```
if task_achieved:    reward = 1.0  (clamped to 0.999 before return)
else:
    reward = partial_progress * 0.8       # base scaled to [0.0, 0.8]
    if progress_increased: reward += 0.1  # dense step signal
    if phishing_missed:    reward  = 0.0  # hard penalty → clamped to 0.001
    reward = clamp(reward, 0.001, 0.999)  # NEVER exactly 0 or 1
    reward *= 0.85 ** hints_used          # hint decay
```

**Phishing guardrail**: if an agent takes any non-`flag_phishing` action on a confirmed phishing email, the episode reward is set to the minimum (0.001). This enforces safety-first behaviour.

---

## Features

- **30 tasks** across Easy / Medium / Hard tiers
- **Thread context**: prior messages included in the observation for follow-up emails
- **Tool registry**: calendar, CRM, ticketing system, escalation paths
- **Phishing guardrails**: safety-first reward shaping
- **Curriculum learning**: mastery tracking, priority-based task selection
- **Hint system**: progressive hints at ×0.85 reward decay each
- **Anti-reward-hacking**: action allowlisting, deduplication, state verification
- **OpenEnv-compatible**: standard HTTP + WebSocket API on port 7860

---

## Quick Start

```python
from client import EmailTriageEnv, EmailTriageAction

with EmailTriageEnv(base_url="http://localhost:7860") as env:
    result = env.reset()
    print(result.observation.email_subject)

    result = env.step(EmailTriageAction(action='{"action": "classify", "priority": "high"}'))
    print(result.reward)
```

---

## Running

### Docker

```bash
docker build -t email-triage-env .
docker run -p 7860:7860 email-triage-env
```

### Local

```bash
pip install -e ".[dev]"
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

---

## Inference

```bash
# Required environment variables
export API_BASE_URL="https://your-litellm-proxy/v1"
export API_KEY="your-key"
export MODEL_NAME="gpt-4o-mini"
export ENV_URL="http://localhost:7860"

python inference.py
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `ENV_URL` | `http://localhost:7860` | Environment server URL |
| `API_BASE_URL` | — | LLM API endpoint (LiteLLM proxy) |
| `API_KEY` | — | API key |
| `MODEL_NAME` | `gpt-4o-mini` | Model identifier |
| `MAX_STEPS` | `10` | Max steps per episode |
| `NUM_EPISODES` | `11` | Episodes to run |

---

## Project Structure

```
email-triage/
├── __init__.py
├── models.py                     # Pydantic data models
├── client.py                     # OpenEnv client
├── inference.py                  # LLM agent inference (submission entry point)
├── openenv.yaml                  # OpenEnv manifest
├── Dockerfile
├── requirements.txt
├── server/
│   ├── app.py                    # FastAPI application (port 7860)
│   ├── email_triage_environment.py  # Core RL environment
│   └── services/
│       ├── tasks.py              # 30 task definitions
│       ├── task_grader.py        # Reward shaping & grading engine
│       ├── curriculum.py         # Curriculum + mastery tracking
│       └── episode_tracker.py    # Per-episode state
└── tests/
    └── test_grader.py
```
