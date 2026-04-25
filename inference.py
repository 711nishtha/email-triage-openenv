"""
inference.py — Email Triage Agent Society
==========================================
Multi-agent colony inference for the Meta PyTorch OpenEnv Hackathon 2026.

Architecture:
  ┌─────────────────────────────────────────┐
  │          AGENT SOCIETY                  │
  │  ┌──────────────┐  ┌─────────────────┐  │
  │  │ TriageAgent  │  │PhishingForensic │  │
  │  └──────┬───────┘  └────────┬────────┘  │
  │         │  Shared Blackboard│           │
  │  ┌──────▼───────┐  ┌────────▼────────┐  │
  │  │SafetyAuditor │  │MemoryConsistency│  │
  │  └──────┬───────┘  └────────┬────────┘  │
  │         └──────┬────────────┘           │
  │         ┌──────▼──────┐                 │
  │         │  Debate     │  ← votes, vetos │
  │         │ Coordinator │                 │
  │         └──────┬──────┘                 │
  │                │ Final action JSON       │
  └────────────────┼────────────────────────┘
                   ▼
             OpenEnv step()

MANDATORY env vars (injected by hackathon LiteLLM proxy):
  API_BASE_URL   LLM API endpoint (never hardcoded)
  API_KEY        API key (never hardcoded)
  MODEL_NAME     Model identifier

Optional env vars:
  ENV_URL        Environment server  (default: http://localhost:7860)
  MAX_STEPS      Steps per episode   (default: 10)
  NUM_EPISODES   Episodes to run     (default: 11)
  SOCIETY_MODE   "full"|"fast"       (default: "full")
                 "fast" = single agent (for latency-constrained proxies)

Stdout format (required by OpenEnv validator):
  [START] task=<id> env=email_triage model=<model>
  [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

All rewards clamped strictly to (0.001, 0.999).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import textwrap
import time
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — ALL from environment, NOTHING hardcoded
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = (
    os.environ.get("API_KEY")
    or os.environ.get("HF_TOKEN")
    or os.environ.get("GROQ_API_KEY")
    or "placeholder"
)
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860")
MAX_STEPS: int = int(os.environ.get("MAX_STEPS", "10"))
NUM_EPISODES: int = int(os.environ.get("NUM_EPISODES", "11"))
SOCIETY_MODE: str = os.environ.get("SOCIETY_MODE", "full")  # "full" | "fast"
BENCHMARK: str = "email_triage"

REWARD_MIN = 0.001
REWARD_MAX = 0.999

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
_log = logging.getLogger("society")

# ---------------------------------------------------------------------------
# LLM client — points at whatever API_BASE_URL is injected
# ---------------------------------------------------------------------------

_llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)


def _call_llm(
    system: str,
    user: str,
    max_tokens: int = 350,
    temperature: float = 0.15,
) -> str:
    """
    Single LLM call through the configured proxy.
    Falls back to a safe default on any error.
    Strips markdown fences automatically.
    """
    try:
        resp = _llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = (resp.choices[0].message.content or "").strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(l for l in lines if not l.startswith("```")).strip()
        return text
    except Exception as exc:
        _log.warning("LLM call failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Shared Blackboard
# ---------------------------------------------------------------------------

class Blackboard:
    """
    Shared memory for agent society. Agents post their analyses here;
    the Debate Coordinator reads all posts to reach consensus.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.triage_vote: str = ""
        self.triage_reasoning: str = ""
        self.phishing_verdict: str = ""           # "phishing" | "clean" | "suspicious"
        self.phishing_confidence: float = 0.0
        self.safety_veto: bool = False
        self.safety_message: str = ""
        self.memory_context: str = ""             # relevant history summary
        self.debate_transcript: list[str] = []
        self.final_action: str = ""
        self.consensus_confidence: float = 0.0

    def post(self, agent: str, message: str) -> None:
        self.debate_transcript.append(f"[{agent}] {message}")

    def summary(self) -> str:
        return "\n".join(self.debate_transcript[-8:])


# ---------------------------------------------------------------------------
# System prompts per agent role
# ---------------------------------------------------------------------------

_TRIAGE_SYSTEM = textwrap.dedent("""
You are the Triage Agent in an enterprise email processing system.
Your job: analyze the email and vote for the best single action.

Output ONLY valid JSON:
{"vote": "<action_json_string>", "priority": "low|medium|high|urgent",
 "category": "billing|support|sales|hr|it|general", "reasoning": "<2 sentences>"}

Action options:
  {"action":"classify","priority":"<p>","category":"<c>"}
  {"action":"assign_queue","queue":"<billing|support|sales|hr|it|general>"}
  {"action":"escalate","to":"<manager|director|vp|on-call>","reason":"<r>"}
  {"action":"reply","tone":"<professional|empathetic|welcoming>","summary":"<s>"}
  {"action":"use_tool","tool":"<calendar|crm|ticketing|notification>","params":{}}
  {"action":"flag_phishing"}

Rules: ONE action. No markdown. If email could be phishing, vote flag_phishing immediately.
""").strip()

_PHISHING_SYSTEM = textwrap.dedent("""
You are the Phishing Forensic Agent. Your ONLY job is to assess whether an email
is a phishing/social-engineering attack.

Check for:
- Lookalike/typosquat domains (paypa1, amaz0n, microsofft, etc.)
- Wire transfer or credential requests
- Urgency manipulation ("act in 2 hours or lose account")
- Mismatched sender name vs domain
- Suspicious URLs (http not https, unusual TLD like .ru .biz .org for corporate)
- CEO/executive impersonation

Output ONLY valid JSON:
{"verdict": "phishing|suspicious|clean",
 "confidence": 0.0_to_1.0,
 "signals": ["signal1","signal2"],
 "recommendation": "flag_phishing|proceed|verify"}
""").strip()

_SAFETY_SYSTEM = textwrap.dedent("""
You are the Safety Auditor Agent. You receive the proposed action from the Triage Agent
and the phishing verdict from the Phishing Forensic Agent.

Your job: VETO if the proposed action violates enterprise security rules.

VETO conditions (mandatory):
1. Phishing verdict is "phishing" AND proposed action is NOT flag_phishing
2. Email body contains wire-transfer language AND proposed action is NOT flag_phishing
3. Email requests credentials/passwords AND proposed action is "reply"
4. Sender domain is lookalike/external AND proposed action is "reply"

Output ONLY valid JSON:
{"veto": true|false, "reason": "<if veto, explain why>",
 "corrected_action": "<if veto, the safe action to take instead>",
 "safe_to_proceed": true|false}
""").strip()

_MEMORY_SYSTEM = textwrap.dedent("""
You are the Memory & Consistency Agent. You see the current email and the thread history.

Your job: summarize the conversation context and flag any inconsistencies.

Output ONLY valid JSON:
{"context_summary": "<1-2 sentences about prior conversation>",
 "consistency_issues": ["issue1"] or [],
 "suggested_priority_adjustment": "none|escalate|downgrade",
 "reasoning": "<why>"}
""").strip()

_DEBATE_SYSTEM = textwrap.dedent("""
You are the Debate Coordinator in an email triage agent society.
You receive analysis from: Triage Agent, Phishing Forensic Agent, Safety Auditor, Memory Agent.

Your job: reach a final decision by weighing all inputs.

HARD RULES (override everything):
1. If Safety Auditor vetoed → use the corrected_action from Safety Auditor
2. If Phishing verdict = "phishing" → output flag_phishing action
3. If phishing confidence > 0.7 → output flag_phishing action

Otherwise: synthesize the best action from the debate.

Output ONLY valid JSON:
{"final_action": "<complete action JSON string, e.g. {\"action\":\"classify\",...}>",
 "confidence": 0.0_to_1.0,
 "decision_reasoning": "<2-3 sentences explaining the consensus>",
 "dissenting_views": "<any agent that disagreed>"}
""").strip()

_FAST_SYSTEM = textwrap.dedent("""
You are an expert enterprise email triage specialist.
Output ONLY one valid JSON action. Include a "reasoning" field.

Actions:
  {"action":"classify","priority":"low|medium|high|urgent","category":"billing|support|sales|hr|it|general"}
  {"action":"assign_queue","queue":"billing|support|sales|hr|it|general"}
  {"action":"flag_phishing"}
  {"action":"escalate","to":"manager|director|vp|on-call","reason":"<reason>"}
  {"action":"reply","tone":"professional|empathetic|welcoming","summary":"<brief>"}
  {"action":"use_tool","tool":"calendar|crm|ticketing|notification","params":{}}

Safety RULES:
- Lookalike domain → flag_phishing immediately
- Wire transfer request → flag_phishing immediately
- Credential request → flag_phishing immediately

Add a "reasoning" field (2 sentences) for bonus reward.
""").strip()


# ---------------------------------------------------------------------------
# Individual agent calls
# ---------------------------------------------------------------------------

def _run_triage_agent(bb: Blackboard, email_ctx: str) -> None:
    user = f"Analyze this email and vote for the best action:\n\n{email_ctx}"
    raw = _call_llm(_TRIAGE_SYSTEM, user, max_tokens=300)
    try:
        parsed = json.loads(raw)
        bb.triage_vote = parsed.get("vote", "")
        bb.triage_reasoning = parsed.get("reasoning", "")
        bb.post("TriageAgent", f"vote={bb.triage_vote[:80]} reason={bb.triage_reasoning[:60]}")
    except (json.JSONDecodeError, ValueError):
        bb.triage_vote = '{"action":"classify","priority":"medium","category":"general"}'
        bb.post("TriageAgent", f"parse_error fallback vote={bb.triage_vote}")


def _run_phishing_agent(bb: Blackboard, email_ctx: str) -> None:
    user = f"Forensically examine this email for phishing signals:\n\n{email_ctx}"
    raw = _call_llm(_PHISHING_SYSTEM, user, max_tokens=250)
    try:
        parsed = json.loads(raw)
        bb.phishing_verdict = parsed.get("verdict", "clean")
        bb.phishing_confidence = float(parsed.get("confidence", 0.0))
        signals = parsed.get("signals", [])
        bb.post(
            "PhishingForensic",
            f"verdict={bb.phishing_verdict} conf={bb.phishing_confidence:.2f} signals={signals[:3]}",
        )
    except (json.JSONDecodeError, ValueError):
        bb.phishing_verdict = "clean"
        bb.phishing_confidence = 0.0
        bb.post("PhishingForensic", "parse_error, defaulting to clean")


def _run_safety_agent(bb: Blackboard, email_ctx: str) -> None:
    user = textwrap.dedent(f"""
    Email context:
    {email_ctx}

    Triage Agent proposed action: {bb.triage_vote}
    Phishing Forensic verdict: {bb.phishing_verdict} (confidence: {bb.phishing_confidence:.2f})

    Should this action be vetoed?
    """).strip()
    raw = _call_llm(_SAFETY_SYSTEM, user, max_tokens=200)
    try:
        parsed = json.loads(raw)
        bb.safety_veto = bool(parsed.get("veto", False))
        bb.safety_message = parsed.get("reason", "")
        if bb.safety_veto:
            corrected = parsed.get("corrected_action", '{"action":"flag_phishing"}')
            bb.triage_vote = corrected  # override with safe action
            bb.post("SafetyAuditor", f"VETO: {bb.safety_message[:80]} → corrected={corrected[:60]}")
        else:
            bb.post("SafetyAuditor", "no veto, action approved")
    except (json.JSONDecodeError, ValueError):
        bb.post("SafetyAuditor", "parse_error, no veto applied")


def _run_memory_agent(bb: Blackboard, email_ctx: str, history: list[str]) -> None:
    history_block = "\n".join(history[-4:]) if history else "No prior actions."
    user = textwrap.dedent(f"""
    Current email:
    {email_ctx}

    Prior actions this episode:
    {history_block}

    Summarize context and flag consistency issues.
    """).strip()
    raw = _call_llm(_MEMORY_SYSTEM, user, max_tokens=200)
    try:
        parsed = json.loads(raw)
        bb.memory_context = parsed.get("context_summary", "")
        issues = parsed.get("consistency_issues", [])
        adj = parsed.get("suggested_priority_adjustment", "none")
        bb.post("MemoryConsistency", f"context={bb.memory_context[:60]} issues={issues} adj={adj}")
    except (json.JSONDecodeError, ValueError):
        bb.memory_context = "No context available."
        bb.post("MemoryConsistency", "parse_error, no context adjustment")


def _run_debate_coordinator(bb: Blackboard, email_ctx: str) -> str:
    """
    Reads the full blackboard debate and produces the final action.
    Hard safety rules are enforced here as a last resort even if agents failed.
    """
    # Hard safety override before calling LLM (pure Python, no LLM cost)
    if bb.phishing_verdict == "phishing" or bb.phishing_confidence >= 0.7 or bb.safety_veto:
        final = '{"action": "flag_phishing"}'
        bb.post("DebateCoordinator", f"HARD_SAFETY_OVERRIDE → {final}")
        return final

    user = textwrap.dedent(f"""
    Email context:
    {email_ctx}

    Debate transcript:
    {bb.summary()}

    Triage vote: {bb.triage_vote}
    Phishing verdict: {bb.phishing_verdict} (conf={bb.phishing_confidence:.2f})
    Safety veto: {bb.safety_veto}
    Memory context: {bb.memory_context}

    Produce the final action decision.
    """).strip()

    raw = _call_llm(_DEBATE_SYSTEM, user, max_tokens=300)
    try:
        parsed = json.loads(raw)
        final = parsed.get("final_action", bb.triage_vote)
        bb.consensus_confidence = float(parsed.get("confidence", 0.5))
        reasoning = parsed.get("decision_reasoning", "")
        bb.post("DebateCoordinator", f"final={str(final)[:80]} conf={bb.consensus_confidence:.2f}")
        # final might be a nested JSON string or a dict
        if isinstance(final, dict):
            return json.dumps(final)
        return str(final)
    except (json.JSONDecodeError, ValueError):
        bb.post("DebateCoordinator", "parse_error, using triage vote")
        return bb.triage_vote or '{"action":"classify","priority":"medium","category":"general"}'


# ---------------------------------------------------------------------------
# Society orchestrator
# ---------------------------------------------------------------------------

class AgentSociety:
    """
    Orchestrates the full multi-agent debate pipeline.

    In SOCIETY_MODE="full"  : runs all 5 agents (5 LLM calls per step)
    In SOCIETY_MODE="fast"  : runs a single unified agent (1 LLM call per step)
    """

    def __init__(self) -> None:
        self._bb = Blackboard()
        self._mode = SOCIETY_MODE

    def deliberate(
        self,
        email_subject: str,
        email_sender: str,
        email_body: str,
        thread_history: list,
        step: int,
        last_result: str,
        last_reward: float,
        history: list[str],
        task_description: str,
    ) -> tuple[str, dict]:
        """
        Run the agent society and return (action_json_string, debate_meta).
        """
        self._bb.reset()

        email_ctx = _build_email_context(
            task_description, email_subject, email_sender,
            email_body, thread_history, step, last_result, last_reward,
        )

        if self._mode == "fast":
            return self._fast_path(email_ctx)

        # Full society pipeline
        _run_triage_agent(self._bb, email_ctx)
        _run_phishing_agent(self._bb, email_ctx)
        _run_safety_agent(self._bb, email_ctx)
        _run_memory_agent(self._bb, email_ctx, history)
        final = _run_debate_coordinator(self._bb, email_ctx)

        # Validate final action is parseable JSON
        final = _validate_action(final)

        meta = {
            "triage_vote": self._bb.triage_vote,
            "phishing": f"{self._bb.phishing_verdict}({self._bb.phishing_confidence:.2f})",
            "safety_veto": self._bb.safety_veto,
            "consensus_confidence": self._bb.consensus_confidence,
            "debate_steps": len(self._bb.debate_transcript),
        }
        return final, meta

    def _fast_path(self, email_ctx: str) -> tuple[str, dict]:
        """Single-agent fallback for latency-sensitive proxies."""
        raw = _call_llm(_FAST_SYSTEM, f"Triage this email:\n\n{email_ctx}", max_tokens=300)
        action = _validate_action(raw)
        return action, {"mode": "fast", "raw": raw[:80]}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_email_context(
    task_description: str,
    subject: str,
    sender: str,
    body: str,
    thread_history: list,
    step: int,
    last_result: str,
    last_reward: float,
) -> str:
    thread_block = ""
    if thread_history:
        parts = []
        for msg in thread_history:
            if hasattr(msg, "sender"):
                parts.append(f"  [{getattr(msg,'timestamp','')}] {msg.sender}: {msg.body[:100]}")
            elif isinstance(msg, dict):
                parts.append(f"  [{msg.get('timestamp','')}] {msg.get('sender','')}: {msg.get('body','')[:100]}")
        thread_block = "\nThread history:\n" + "\n".join(parts)

    return textwrap.dedent(f"""
    TASK: {task_description}

    Email:
      From: {sender}
      Subject: {subject}
      Body: {body}
    {thread_block}

    Step: {step} | Last reward: {last_reward:.3f} | Last result: {last_result!r}
    """).strip()


def _validate_action(raw: str) -> str:
    """
    Ensure the action is a valid JSON string parseable by the environment.
    Falls back to a safe classify action.
    """
    raw = raw.strip()
    # If it looks like JSON directly
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            if "action" in parsed:
                return raw
        except (json.JSONDecodeError, ValueError):
            pass
    # Try extracting JSON from surrounding text
    import re
    match = re.search(r'\{[^{}]*"action"[^{}]*\}', raw)
    if match:
        candidate = match.group(0)
        try:
            parsed = json.loads(candidate)
            if "action" in parsed:
                return candidate
        except (json.JSONDecodeError, ValueError):
            pass
    # Safe fallback
    return '{"action": "classify", "priority": "medium", "category": "general"}'


def _clamp(v: float) -> float:
    return max(REWARD_MIN, min(REWARD_MAX, v))


# ---------------------------------------------------------------------------
# Logging helpers (strict validator format)
# ---------------------------------------------------------------------------

def _log_start(task_id: str) -> None:
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def _log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_task(env_url: str) -> None:
    """Run NUM_EPISODES episodes against the OpenEnv server at env_url."""
    from client import EmailTriageEnv
    from models import EmailTriageAction, EmailTriageObservation

    society = AgentSociety()

    with EmailTriageEnv(base_url=env_url).sync() as env:
        for episode_idx in range(NUM_EPISODES):
            result = env.reset()
            obs: EmailTriageObservation = result.observation

            task_id = str(obs.task.task_id) if obs.task else f"ep{episode_idx}"
            task_description = obs.task.description if obs.task else ""
            email_subject = obs.task.email_subject if obs.task else ""
            email_sender = obs.task.email_sender if obs.task else ""
            email_body = obs.task.email_body if obs.task else ""
            thread_history = list(obs.task.thread_history) if obs.task else []

            _log_start(task_id)

            history: list[str] = []
            rewards: list[float] = []
            last_result = obs.last_action_result
            last_reward = 0.0
            steps = 0
            success = False

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                # --- Agent Society deliberates ---
                action_str, meta = society.deliberate(
                    email_subject=email_subject,
                    email_sender=email_sender,
                    email_body=email_body,
                    thread_history=thread_history,
                    step=step,
                    last_result=last_result,
                    last_reward=last_reward,
                    history=history,
                    task_description=task_description,
                )

                # Log debug metadata to stderr (not stdout — validator reads stdout only)
                _log.debug(
                    "Episode=%d Step=%d meta=%s action=%s",
                    episode_idx, step, meta, action_str[:80],
                )

                result = env.step(EmailTriageAction(action=action_str))
                obs = result.observation

                raw_reward = obs.reward if obs.reward is not None else 0.0
                done = result.done
                last_result = obs.last_action_result
                last_reward = raw_reward

                # Clamp strictly to (REWARD_MIN, REWARD_MAX) — NEVER exactly 0 or 1
                reward = _clamp(raw_reward)
                rewards.append(reward)
                steps = step

                error_str = None if obs.last_action_valid else obs.last_action_result
                _log_step(step, action_str, reward, done, error_str)

                history.append(
                    f"Step {step}: {action_str} → reward={reward:.3f} | {last_result[:60]}"
                )

                if obs.task_achieved:
                    success = True
                    break

                if done:
                    break

            score = max(rewards) if rewards else REWARD_MIN
            score = _clamp(score)
            _log_end(success=success, steps=steps, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_task(ENV_URL)
