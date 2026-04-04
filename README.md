# Advanced Enterprise Email Triage OpenEnv

An OpenEnv-compliant environment for evaluating AI agents on enterprise email triage tasks.

## Overview

This environment simulates an enterprise inbox where an agent must classify, prioritize, and route emails correctly. It tests real-world skills like phishing detection, urgency assessment, and delegation logic.

## Features

- **3 Task Difficulties**: Easy (1–3 emails), Medium (5–8 emails), Hard (10+ emails)
- **Dense Reward Function**: Partial credit for category, priority, and routing
- **Phishing Traps**: Tests security awareness
- **Streamlit UI**: Optional visual dashboard on port 8501
- **FastAPI Backend**: Always available on port 7860

## Project Structure

```
.
├── README.md
├── openenv.yaml
├── requirements.txt
├── models.py          # Pydantic models
├── data.py            # Synthetic email generation
├── tools.py           # Agent tools
├── graders.py         # Deterministic graders
├── rewards.py         # Reward computation
├── environment.py     # Core environment logic
├── server/
│   └── app.py         # FastAPI server
├── inference.py       # Inference script
├── Dockerfile
└── start.sh
```

## API Endpoints

| Method | Path     | Description                  |
|--------|----------|------------------------------|
| POST   | /reset   | Reset environment, get obs   |
| POST   | /step    | Submit action, get reward    |
| GET    | /state   | Get full environment state   |
| GET    | /health  | Health check                 |

## Action Space

```json
{
  "action_type": "triage | escalate | use_tool | done",
  "email_id": "string",
  "priority": "urgent | high | medium | low",
  "category": "phishing | urgent_business | internal_task | marketing | hr | legal | it_support | finance | spam",
  "route_to": "security | executive | manager | it | hr | finance | archive | trash",
  "tool_name": "string (optional)",
  "tool_args": {}
}
```

## Reward Structure

- Correct category: **+0.4**
- Correct priority: **+0.3**
- Correct routing: **+0.3**
- Missing urgent email: **-0.3 penalty**
- Misclassifying phishing: **-0.4 penalty**
- All rewards clamped to **[0.0, 1.0]**

## Tasks

### Easy
- 2 emails with clear signals
- Suitable for baseline agent testing

### Medium
- 6 emails with mixed priorities
- Some ambiguous cases

### Hard
- 12 emails including phishing traps, urgent CEO requests, and subtle spam
- Tests full triage capability

## Running Locally

```bash
docker build -t email-triage-openenv .
docker run -p 7860:7860 -p 8501:8501 email-triage-openenv
```

## Environment Variables (inference.py)

| Variable      | Description                        |
|---------------|------------------------------------|
| API_BASE_URL  | LLM API base (e.g., Groq endpoint) |
| MODEL_NAME    | Model identifier                   |
| HF_TOKEN      | Hugging Face token (if needed)     |
