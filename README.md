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

---

## Motivation — The Real Enterprise Gap

Every large organisation loses thousands of productive hours each year to poor email triage. Inboxes receive **urgent board-level escalations alongside routine team check-ins, vendor spam, and — increasingly — sophisticated phishing attacks** that impersonate trusted senders. Human triagers suffer from cognitive overload, inconsistency across shifts, and zero memory of prior thread context.

This OpenEnv environment simulates a **production-grade enterprise mailbox** that an AI agent must triage correctly, safely, and efficiently. It captures:

- **Priority discrimination** — distinguishing a CEO security incident from a newsletter
- **Thread-aware reasoning** — understanding email chains and prior commitments
- **Sender importance scoring** — dynamic VIP/domain reputation weighting
- **Phishing safety guardrails** — detecting spoofed domains, urgency manipulation, credential harvesting
- **Business-impact routing** — knowing when to escalate vs. delegate vs. archive
- **Efficiency bonuses** — fewer tool calls = higher score, rewarding focused reasoning

This directly addresses the gap between generic LLM assistants and reliable, auditable enterprise automation.

---

## Project Structure

```
advanced-enterprise-email-triage/
├── README.md
├── .env.example
├── openenv.yaml
├── requirements.txt
├── models.py           # Pydantic typed models
├── data.py             # Synthetic email generator
├── tools.py            # Mock safe tools (calendar, KB, sender-lookup)
├── graders.py          # 3 task graders (deterministic scoring)
├── rewards.py          # Dense reward shaping logic
├── environment.py      # Core OpenEnv environment (reset/step/state)
├── inference.py        # OpenAI agent runner (exact hackathon log format)
├── Dockerfile
├── .dockerignore
├── ui_streamlit/
│   ├── app.py          # Streamlit UI
│   └── requirements.txt
└── server/
    └── app.py          # FastAPI server exposing /reset, /step, /state
```

---

## Interactive User Interface

A professional Streamlit UI is provided for manual triage, performance monitoring, and environment testing.

### Features
- **Task Selection**: Switch between Easy, Medium, and Hard tasks.
- **Email Viewer**: Rich display of email content, thread history, and sender profiles.
- **Action Panel**: Perform triage actions (Categorize, Route, Flag, Respond, Escalate, Ignore).
- **Live Feedback**: Real-time reward signals and grader feedback.
- **Tool Integration**: Trigger mock tools directly from the UI.

---

## Action & Observation Spaces

### Observation

```python
class EmailObservation(BaseModel):
    task_id: str                     # Unique task identifier
    step: int                        # Current step within episode
    max_steps: int                   # Episode budget
    inbox: List[EmailMessage]        # Emails to triage (1–10 per task)
    thread_history: Dict[str, List[EmailMessage]]  # thread_id → prior emails
    sender_profiles: Dict[str, SenderProfile]      # sender → importance metadata
    available_tools: List[str]       # Tools the agent may call this step
    current_score: float             # Cumulative reward so far
    warnings: List[str]              # Safety flags already triggered
    done: bool
```

### Action

```python
class TriageAction(BaseModel):
    action_type: Literal["triage", "use_tool", "escalate", "done"]
    
    # For action_type == "triage"
    email_id: Optional[str]
    priority: Optional[Literal["critical", "high", "medium", "low", "spam"]]
    category: Optional[Literal[
        "security_incident", "executive_request", "hr_matter",
        "vendor_contract", "it_support", "team_update",
        "customer_escalation", "phishing", "newsletter", "other"
    ]]
    route_to: Optional[str]          # team/person to route to
    reasoning: Optional[str]         # agent's chain-of-thought

    # For action_type == "use_tool"
    tool_name: Optional[str]
    tool_params: Optional[Dict[str, Any]]

    # For action_type == "escalate"
    escalation_target: Optional[str]
    escalation_reason: Optional[str]
```

---

## Three Tasks — Difficulty & Description

### Task 1 — Easy: "Morning Inbox Clear"

**Scenario:** A single clear, unambiguous email arrives from a known internal sender. The subject line, body, and sender domain all give consistent signals. No thread history. No tools needed.

**Agent goal:** Correctly classify the priority and category, then route to the right team.

**Difficulty factors:**
- No ambiguity
- No phishing elements
- No tool use required
- Single email only

**Grader:** Exact match on priority (30 pts), category (40 pts), route (30 pts). Partial credit for priority adjacency.

**Baseline score:** `0.72` (GPT-4o-mini zero-shot)

---

### Task 2 — Medium: "Batch Triage with Thread Context"

**Scenario:** Five emails arrive simultaneously. Two are continuations of existing threads (with prior context). One requires a calendar tool check before routing. One is a vendor contract follow-up needing KB lookup.

**Agent goal:** Triage all five emails correctly, use tools appropriately (not excessively), and preserve thread context in routing decisions.

**Difficulty factors:**
- Thread-aware reasoning required
- Tool use needed (calendar + KB)
- Batch ordering matters (urgent first)
- Sender reputation scoring influences priority

**Grader:** Per-email scoring (category 25%, priority 25%, route 25%) + tool efficiency bonus (25%). Penalty if urgent email processed last.

**Baseline score:** `0.54` (GPT-4o-mini zero-shot)

---

### Task 3 — Hard: "Threat Inbox — Phishing, Urgency, Ambiguity"

**Scenario:** Eight emails in a mixed inbox. Two are sophisticated phishing attempts (spoofed CEO domain, fake IT password-reset). One is a genuine critical security incident. One is ambiguous (external board member making an unusual wire-transfer request — classic BEC pattern). Three are legitimate-but-noisy urgent requests. One is a harmless newsletter that mimics urgent formatting.

**Agent goal:** Correctly identify all phishing/BEC emails, escalate the real security incident, avoid false-positive security escalations on legitimate urgent mail, and process the rest with accurate priority/category/routing.

**Difficulty factors:**
- Phishing detection (spoofed domains, urgency manipulation, credential requests)
- BEC (Business Email Compromise) recognition
- False positive avoidance on legitimate urgency
- Safety escalation path must be used correctly
- Multiple conflicting urgency signals

**Grader:**
- Phishing detection: F1 score over phishing labels (40 pts)
- Security incident escalation: binary correct/miss (20 pts)  
- Legitimate email accuracy: per-email F1 (30 pts)
- Safety compliance: penalty if credentials requested and not flagged (−30 pts), bonus for correct BEC identification (+10 pts)

**Baseline score:** `0.31` (GPT-4o-mini zero-shot)

---

## Reward Function — Dense Partial Signals

| Signal | Reward |
|---|---|
| Correct category | +0.15 per email |
| Correct priority | +0.10 per email |
| Correct route | +0.10 per email |
| Phishing correctly flagged | +0.25 per email |
| Security incident escalated | +0.30 |
| BEC pattern flagged | +0.20 |
| Tool used correctly | +0.05 per call |
| Tool used unnecessarily | −0.05 per call |
| Critical email missed (priority=low/spam) | −0.40 |
| Phishing email not flagged | −0.30 |
| False positive on legitimate exec email | −0.20 |
| Credentials clicked / not flagged | −0.50 |
| Episode completed within step budget | +0.10 |
| Wrong escalation target | −0.15 |

Total score normalised to **[0.0, 1.0]** per episode.

---

## Setup — Local Testing

### Prerequisites

- Python 3.11+
- Docker (for containerised runs)

### 1. Clone and install

```bash
git clone https://github.com/711nishtha/advanced-enterprise-email-triage
cd advanced-enterprise-email-triage
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
# Edit .env — fill in your real values (OpenAI, Groq, or Gemini)
# The inference agent automatically loads these via python-dotenv.
```

#### Recommended Free Provider: Groq
For the best free performance, we recommend using [Groq](https://console.groq.com/keys):
- **API_BASE_URL**: `https://api.groq.com/openai/v1`
- **MODEL_NAME**: `llama-3.3-70b-versatile`

### 3. Run the FastAPI server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### 4. Run the inference agent

```bash
python inference.py --task easy
python inference.py --task medium
python inference.py --task hard
```

### 5. Run the Streamlit UI

```bash
# In a new terminal (while backend is running)
pip install -r ui_streamlit/requirements.txt
streamlit run ui_streamlit/app.py
```

### 6. Validate OpenEnv spec

```bash
openenv validate openenv.yaml
```

---

## Docker Build & Run

```bash
# Build
docker build -t email-triage-env .

# Run (pass env vars — never bake into image)
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e API_BASE_URL=$API_BASE_URL \
  -e MODEL_NAME=$MODEL_NAME \
  email-triage-env
```

---

## HF Spaces Deployment

1. Create a new **Docker** Space on Hugging Face
2. Push this repository to the Space
3. Set the following **Secrets** (not variables) in Space Settings:
   - `OPENAI_API_KEY`
   - `API_BASE_URL` (optional — defaults to OpenAI)
   - `MODEL_NAME` (optional — defaults to `gpt-4o-mini`)
   - `HF_TOKEN` (if private model access needed)
4. The Space auto-builds from `Dockerfile` and exposes port `7860`
5. `/reset`, `/step`, `/state` endpoints become available at your Space URL

---

## Deploying the UI to HF Spaces

The Streamlit UI can be deployed as a separate **Streamlit** Space:

1. Create a new **Streamlit** Space on Hugging Face
2. Upload the `ui_streamlit/` folder contents or point to it
3. Set the **Secret** `BACKEND_URL` in Space Settings to your Backend Space URL (e.g., `https://your-backend.hf.space`)
4. The UI will automatically connect to your environment API.

---

## Judging Criteria Alignment

| Criterion | Weight | How This Env Scores High |
|---|---|---|
| **Utility** | 30% | Real enterprise pain point; plug-in ready for corporate IT agents |
| **Task Quality** | 25% | 3 well-separated difficulty levels; deterministic graders; dense rewards |
| **Technical Depth** | 20% | Thread-aware context, BEC detection, dynamic sender scoring |
| **Creativity** | 15% | Phishing F1 grader, BEC heuristics, step-budget efficiency bonus |
| **Documentation** | 10% | This README; inline docstrings; `.env.example`; security section |

---

## Security Practices

> ⚠️ **All credentials are loaded exclusively via environment variables. No secrets are ever hardcoded.**

- `os.getenv()` used throughout — never string literals for keys
- `.env.example` contains only placeholder values (`your_key_here`)
- `.dockerignore` excludes `.env`, `*.pem`, `*.key`, `secrets/`
- `Dockerfile` has no `COPY .env` instruction
- `inference.py` exits with a clear error message if required env vars are absent
- No logging of actual credential values — only presence checks are logged

---

## Baseline Scores (Groq Llama-3.3-70b)

| Task | Model | Score | Status |
|---|---|---|---|
| Easy | Llama-3.3-70b-versatile | 0.795 | ✓ PASS |
| Medium | Llama-3.3-70b-versatile | 0.847 | ✓ PASS |
| Hard | Llama-3.3-70b-versatile | 0.787 | ✓ PASS |

---

## License

MIT — see `LICENSE` for details.

---

*Built for Meta PyTorch OpenEnv Hackathon 2026 by a solo developer. All synthetic email data is fictional. No real PII is used.*
