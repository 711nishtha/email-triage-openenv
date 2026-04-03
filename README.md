---
title: Advanced Enterprise Email Triage
emoji: 📧
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Advanced-Enterprise-Email-Triage-OpenEnv

> **Meta PyTorch OpenEnv Hackathon 2026 — Scaler Round 1 Submission**

[![OpenEnv Compatible](https://img.shields.io/badge/OpenEnv-Compatible-blue)](https://github.com/openenv)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-green)](https://python.org)
[![Docker](https://img.shields.io/badge/docker-ready-blue)](https://docker.com)

---

## Deployment & Links

- **GitHub Repository**: [https://github.com/711nishtha/email-triage-openenv](https://github.com/711nishtha/email-triage-openenv)
- **Hugging Face Space**: [https://huggingface.co/spaces/nishtha711/email-triage-openenv](https://huggingface.co/spaces/nishtha711/email-triage-openenv)
- **Live Environment API**: `https://nishtha711-email-triage-openenv.hf.space`
- **Interactive UI**: `https://huggingface.co/spaces/nishtha711/email-triage-openenv` (Built with Streamlit)

---

## Platform Overview — Now with Interactive Triage

This platform provides a production-grade enterprise mailbox environment for AI agents and human triagers. It simulates the high-stakes environment of a corporate inbox, complete with:

- **Hybrid Architecture**: A robust FastAPI backend exposing the OpenEnv spec, paired with a beautiful Streamlit UI for manual testing and monitoring.
- **Sophisticated Scenarios**: Three tasks (Easy, Medium, Hard) covering everything from routine IT requests to sophisticated Business Email Compromise (BEC) and phishing attacks.
- **Deep Context**: Thread-aware reasoning, sender reputation scoring, and mock enterprise tools (Calendar, Knowledge Base, Sender Lookup).
- **Dense Rewards**: Real-time feedback signals to guide agent learning and human performance.

---

## Project Structure

```
advanced-enterprise-email-triage/
├── ui_streamlit/
│   ├── app.py          # Professional Streamlit UI
│   └── requirements.txt
├── server/
│   └── app.py          # FastAPI server (OpenEnv API)
├── models.py           # Shared Pydantic schemas
├── data.py             # Synthetic email/profile generator
├── tools.py            # Mock enterprise tools
├── graders.py          # Deterministic task graders
├── rewards.py          # Reward shaping & signals
├── environment.py      # Core OpenEnv implementation
├── inference.py        # OpenAI-compatible agent runner
├── USER_MANUAL.md      # Detailed UI guide
├── Dockerfile          # Multi-process orchestration
├── run_all.sh          # Local runner (Backend + UI)
└── start.sh            # Docker entrypoint script
```

---

## Interactive User Interface

The platform now includes a professional interface for manual triage, agent monitoring, and environment debugging.

- **User Manual**: See [USER_MANUAL.md](./USER_MANUAL.md) for a comprehensive guide on UI features.
- **Task Selection**: Seamlessly switch between difficulty tiers.
- **Security Guardrails**: Visual warnings for suspicious senders and phishing indicators.
- **Live Feedback**: Real-time display of rewards and grader signals.

---

## Action & Observation Spaces

### Observation

```python
class EmailObservation(BaseModel):
    task_id: str                     # Unique task identifier (easy, medium, hard)
    step: int                        # Current step within episode
    max_steps: int                   # Episode budget
    inbox: List[EmailMessage]        # Emails pending triage
    thread_history: Dict[str, List[EmailMessage]]  # thread_id → context
    sender_profiles: Dict[str, SenderProfile]      # sender → reputation/VIP status
    available_tools: List[str]       # Tools allowed for this task
    current_score: float             # Cumulative reward (normalized)
    warnings: List[str]              # Security flags triggered
    done: bool
```

### Action

```python
class TriageAction(BaseModel):
    action_type: Literal["triage", "use_tool", "escalate", "done"]
    
    # triage: priority, category, route_to, reasoning
    # use_tool: tool_name, tool_params
    # escalate: escalation_target, escalation_reason
```

---

## Three Tasks — Difficulty & Description

### Task 1 — Easy: "Morning Inbox Clear"
**Goal:** Basic classification and routing of unambiguous internal emails.

### Task 2 — Medium: "Batch Triage with Thread Context"
**Goal:** Manage 5 emails using thread context, calendar checks, and KB lookups.

### Task 3 — Hard: "Threat Inbox — Phishing & BEC"
**Goal:** Detect sophisticated phishing, handle Business Email Compromise, and escalate real security incidents while ignoring newsletters.

---

## Setup — Local Testing

### 1. Clone and Install
```bash
git clone https://github.com/711nishtha/email-triage-openenv
cd email-triage-openenv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Add your OPENAI_API_KEY (or Groq/Gemini key)
```

### 3. Run Everything (Local)
The simplest way to start both the backend and UI:
```bash
./run_all.sh
```
- **UI**: `http://localhost:8501`
- **API**: `http://localhost:7860`

### 4. Run Inference Agent
```bash
python inference.py --task all --base-url http://localhost:7860
```

---

## Docker & HF Spaces

### Docker Run (Local)
The Docker image runs both the FastAPI backend (port 8000) and Streamlit (port 7860).
```bash
docker build -t email-triage-app .
docker run -p 7860:7860 -p 8000:8000 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  email-triage-app
```

### HF Spaces Deployment
1. Push to a **Docker** Space on Hugging Face.
2. The `Dockerfile` automatically sets up the hybrid environment.
3. Access the interactive UI directly on the Space's main page.

---

## Security & Compliance
- **Zero Secrets Hardcoded**: All credentials loaded via environment variables.
- **PII Safe**: All email data is synthetically generated; no real user data is used.
- **Non-Root Runtime**: Docker container runs as `appuser` for enhanced security.

---

## Baseline Scores (Groq Llama-3.3-70b)

| Task | Score | Status |
|---|---|---|
| Easy | 0.7950 | ✓ PASS |
| Medium | 0.8316 | ✓ PASS |
| Hard | 0.7897 | ✓ PASS |

---

## License
MIT — see `LICENSE` for details.
