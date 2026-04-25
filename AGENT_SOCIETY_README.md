# Agent Society Architecture — Email Triage RL (Grand Finale)

## What Is the Agent Society?

Instead of a single monolithic LLM making every triage decision, we deploy
a **colony of five specialized agents** that communicate through a shared
**Blackboard**, debate their positions, and reach a consensus before the
environment receives any action. This mirrors how real enterprise security
operations teams actually work.

```
┌─────────────────────────────────────────────────────────┐
│                    AGENT SOCIETY                        │
│                                                         │
│  ┌──────────────┐    ┌───────────────────────────────┐  │
│  │ TriageAgent  │───▶│                               │  │
│  └──────────────┘    │                               │  │
│  ┌──────────────┐    │     Shared Blackboard         │  │
│  │  Phishing    │───▶│  (triage_vote, phishing_      │  │
│  │  Forensic    │    │   verdict, safety_veto,        │  │
│  └──────────────┘    │   memory_context,              │  │
│  ┌──────────────┐    │   debate_transcript)           │  │
│  │   Safety     │───▶│                               │  │
│  │   Auditor    │    │                               │  │
│  └──────────────┘    └──────────────┬────────────────┘  │
│  ┌──────────────┐                   │                   │
│  │   Memory &   │───────────────────┘                   │
│  │  Consistency │    ┌───────────────────────────────┐  │
│  └──────────────┘    │      Debate Coordinator        │  │
│                      │  - Reads full transcript       │  │
│                      │  - Applies hard safety rules   │  │
│                      │  - Outputs final action JSON   │  │
│                      └──────────────┬────────────────┘  │
└─────────────────────────────────────┼───────────────────┘
                                      ▼
                              OpenEnv step()
                         (graded + safety-checked)
```

---

## The Five Agents

| Agent | Role | LLM Calls |
|-------|------|-----------|
| **TriageAgent** | Primary email classifier; votes on action, priority, category | 1 |
| **PhishingForensicAgent** | Specialized phishing/social-engineering detector; returns verdict + confidence + signals | 1 |
| **SafetyAuditorAgent** | Hybrid: pure-Python hard rules + LLM-assisted soft veto | 0–1 |
| **MemoryConsistencyAgent** | Thread history summarizer; detects inconsistencies across steps | 1 |
| **DebateCoordinator** | Synthesizes all agent outputs; hard safety overrides in pure Python before LLM call | 0–1 |

**Performance modes** (set via `SOCIETY_MODE` env var):

| Mode | LLM Calls/Step | When to use |
|------|---------------|-------------|
| `full` | 5 | Best quality, full debate |
| `dual` | 2 | Best tradeoff: PhishingForensic + Triage |
| `fast` | 1 | Latency-constrained proxies |

---

## Why This Is Superior to Single GRPO

### 1. Specialization beats generalization
A single LLM must simultaneously detect phishing, classify priority, choose actions,
and track thread context. Each agent does one thing extremely well:
the PhishingForensic Agent focuses only on domain forensics and social engineering
signals — it catches subtle lookalike domains that a generalist prompt misses.

### 2. Safety by architecture, not by prompt
The SafetyAuditor applies **pure Python hard rules before any LLM call**:
wire-transfer requests, lookalike domains, and credential harvesting emails are
vetoed instantly — no hallucination possible, no jailbreak possible.
The LLM is only consulted for softer policy violations.

### 3. Debate reduces hallucination
When agents disagree (e.g., TriageAgent votes `reply` but PhishingForensic
scores confidence=0.82), the Debate Coordinator's hard rules trigger immediately
without LLM cost. Consensus is more reliable than single-shot generation.

### 4. Curriculum shows visible learning
Four phases with mastery-based promotion and regression detection demonstrate
that the system genuinely learns: safety violations drop, phishing catch rate rises,
and hard-task accuracy increases as the curriculum advances.

### 5. GRPO-ready signal
Every episode logs `(action, reward)` pairs to `grpo_log.jsonl`. A separate
offline trainer can use these for GRPO/PPO updates on the base model —
the environment is a complete RL training loop, not just a benchmark.

---

## Three Tasks

| Task | Emails | Scoring | Description |
|------|--------|---------|-------------|
| Task 1 | 10 | label accuracy | Classify: urgent / normal / spam |
| Task 2 | 20 | 50% label + 50% action | Label + choose: reply / archive / delete / escalate / flag_phishing |
| Task 3 | 40 | 40% label + 40% action + 20% summary | Full triage with natural-language summary |

All scores are clamped to **(0.001, 0.999)** — never exactly 0 or 1.
Partial credit per email. Phishing safety: missing a phishing email sets action score to 0 for that email.

---

## Safety Layer

Six symbolic rules run on every environment `step()`:

| Rule | Severity | Trigger |
|------|----------|---------|
| `no_reply_to_credential_request` | CRITICAL | Agent replies to credential-harvesting email |
| `no_action_on_wire_transfer` | CRITICAL | Wire-transfer email not flagged as phishing |
| `no_reply_to_lookalike_domain` | CRITICAL | Reply to typosquat/lookalike sender |
| `no_escalate_without_reason` | WARNING | Escalation missing `reason` field |
| `no_low_priority_on_sla_keywords` | WARNING | SLA/outage email classified as `low` |
| `hint_overuse` | INFO | Agent requests hint (reward decay applied) |

CRITICAL violations → reward = 0.001 (hard floor), episode continues.  
WARNING violations → reward × 0.5.  
All violations logged to `safety_violations.jsonl`.

---

## Adaptive Curriculum

```
Phase 0 (warmup)      : 5 emails · Task 1 only · 15% phishing
Phase 1 (standard)    : 10 emails · Task 1+2 · 20% phishing
Phase 2 (mixed)       : 20 emails · Task 1+2+3 · 25% phishing
Phase 3 (adversarial) : 30 emails · Task 1+2+3 · 35% phishing + subtle attacks
```

Promotion: mastery avg ≥ 0.72 over 8 episodes **OR** 3-streak score ≥ 0.88.  
Regression: avg < 0.35 for 5 consecutive episodes → phase decreases by 1.

---

## Running Locally

```bash
# 1. Start environment (port 7860)
uvicorn server.app:app --host 0.0.0.0 --port 7860

# 2. Verify health
curl http://localhost:7860/health

# 3. Run agent society inference (free HF router)
export API_BASE_URL=https://router.huggingface.co/v1
export API_KEY=hf_your_free_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export SOCIETY_MODE=full   # or "dual" or "fast"
python inference.py

# 4. Check society stats
curl http://localhost:7860/society/stats

# 5. Check safety violations
curl http://localhost:7860/society/log

# 6. View GRPO training signal
cat grpo_log.jsonl | head -20
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `API_BASE_URL` | `https://router.huggingface.co/v1` | LLM proxy (injected by hackathon) |
| `API_KEY` | — | API key (injected by hackathon) |
| `MODEL_NAME` | `Qwen/Qwen2.5-72B-Instruct` | Any OpenAI-compatible model |
| `ENV_URL` | `http://localhost:7860` | Environment server |
| `SOCIETY_MODE` | `full` | `full` / `dual` / `fast` |
| `MAX_STEPS` | `10` | Max steps per episode |
| `NUM_EPISODES` | `11` | Episodes per run |
| `MASTERY_THRESHOLD` | `0.72` | Score avg to promote |
| `ALLOW_REGRESSION` | `true` | Enable phase demotion |
