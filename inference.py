"""
inference.py — Email Triage Agent Society (Meta PyTorch OpenEnv Hackathon 2026)
================================================================================
Multi-agent colony inference for the OpenEnv email-triage benchmark.

Architecture: 5-agent society with Shared Blackboard and real-time debate.
  TriageAgent → PhishingForensicAgent → SafetyAuditorAgent → MemoryAgent
  → DebateCoordinator → final action JSON → OpenEnv step()

MANDATORY env vars (injected by hackathon LiteLLM proxy):
  API_BASE_URL   LLM API endpoint (NEVER hardcoded)
  API_KEY        API key (NEVER hardcoded)
  MODEL_NAME     Model identifier (default: Qwen/Qwen2.5-72B-Instruct)

Optional env vars:
  ENV_URL        Environment server  (default: http://localhost:7860)
  MAX_STEPS      Steps per episode   (default: 10)
  NUM_EPISODES   Episodes to run     (default: 11)
  SOCIETY_MODE   "full"|"dual"|"fast" (default: "full")
  GROQ_API_KEY   Fallback for local testing (only used if API_KEY not set)

Stdout format (required by OpenEnv validator — exact format):
  [START] task=<id> env=email_triage model=<model>
  [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

All rewards clamped strictly to (0.001, 0.999) — never exactly 0.0 or 1.0.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import textwrap
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration — ALL from environment variables, NOTHING hardcoded
# ---------------------------------------------------------------------------

# Primary: injected by hackathon LiteLLM proxy
API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY: str = (
    os.environ.get("API_KEY")
    or os.environ.get("HF_TOKEN")
    or os.environ.get("GROQ_API_KEY")  # local testing fallback only
    or "placeholder"
)
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
ENV_URL: str = os.environ.get("ENV_URL", "http://localhost:7860")
MAX_STEPS: int = int(os.environ.get("MAX_STEPS", "10"))
NUM_EPISODES: int = int(os.environ.get("NUM_EPISODES", "11"))
SOCIETY_MODE: str = os.environ.get("SOCIETY_MODE", "full")  # "full" | "dual" | "fast"
BENCHMARK: str = "email_triage"

REWARD_MIN = 0.001
REWARD_MAX = 0.999

# Stderr logging only — validator reads stdout exclusively
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
_log = logging.getLogger("inference")

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
    Falls back to an empty string on any error (agents handle fallback).
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
        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(ln for ln in lines if not ln.startswith("```")).strip()
        return text
    except Exception as exc:
        _log.warning("LLM call failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Shared Blackboard
# ---------------------------------------------------------------------------


class Blackboard:
    """
    Shared memory for the agent colony.

    Agents post their structured analyses here. The Debate Coordinator
    reads the full transcript to produce a consensus action.

    Communication pattern:
      Each agent reads prior posts → computes its analysis → posts result.
      The Debate Coordinator reads the complete transcript last.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # TriageAgent output
        self.triage_action: str = ""
        self.triage_priority: str = "medium"
        self.triage_category: str = "general"
        self.triage_reasoning: str = ""

        # PhishingForensicAgent output
        self.phishing_verdict: str = "clean"       # "phishing"|"suspicious"|"clean"
        self.phishing_confidence: float = 0.0
        self.phishing_signals: list[str] = []

        # SafetyAuditorAgent output
        self.safety_veto: bool = False
        self.safety_corrected_action: str = ""
        self.safety_message: str = ""

        # MemoryConsistencyAgent output
        self.memory_context: str = ""
        self.memory_priority_adjustment: str = "none"

        # Debate transcript (append-only)
        self.transcript: list[str] = []

        # Final output from DebateCoordinator
        self.final_action: str = ""
        self.consensus_confidence: float = 0.5
        self.decision_reasoning: str = ""

    def post(self, agent: str, msg: str) -> None:
        """Append a message to the debate transcript."""
        self.transcript.append(f"[{agent}] {msg}")

    def debate_log(self, max_lines: int = 10) -> str:
        return "\n".join(self.transcript[-max_lines:])


# ---------------------------------------------------------------------------
# System prompts (concise to minimize token cost)
# ---------------------------------------------------------------------------

_SYS_TRIAGE = textwrap.dedent("""
You are the Triage Agent in an enterprise email processing system.
Classify the email and vote for the best single action.
Output ONLY valid JSON (no markdown, no explanation):
{"action_json": "<action string>", "priority": "low|medium|high|urgent",
 "category": "billing|support|sales|hr|it|general",
 "reasoning": "<2 sentences max>"}

Valid action strings (choose ONE):
  {"action":"classify","priority":"<p>","category":"<c>"}
  {"action":"assign_queue","queue":"<billing|support|sales|hr|it|general>"}
  {"action":"escalate","to":"<manager|director|vp|on-call>","reason":"<r>"}
  {"action":"reply","tone":"<professional|empathetic|welcoming>","summary":"<s>"}
  {"action":"use_tool","tool":"<calendar|crm|ticketing|notification>","params":{}}
  {"action":"flag_phishing"}

RULE: If any phishing signal is visible → vote flag_phishing immediately.
""").strip()

_SYS_PHISHING = textwrap.dedent("""
You are the Phishing Forensic Agent. Analyze ONLY for phishing/social-engineering.
Output ONLY valid JSON:
{"verdict": "phishing|suspicious|clean",
 "confidence": 0.00_to_1.00,
 "signals": ["signal1","signal2"],
 "recommendation": "flag_phishing|proceed|verify"}

Phishing signals to check:
- Lookalike/typosquat domain (paypa1, amaz0n, microsofft, -secure-msg., .ru, .biz)
- Wire transfer or bank routing number requests
- Credential/password requests via email
- Urgency manipulation ("act now or lose access in 2 hours")
- CEO/executive impersonation from external domain
- Mismatched sender name vs domain
- Suspicious URLs (non-HTTPS, unusual TLD)
""").strip()

_SYS_SAFETY = textwrap.dedent("""
You are the Safety Auditor. Decide if the proposed action must be vetoed.
Output ONLY valid JSON:
{"veto": true|false,
 "reason": "<why, if veto>",
 "corrected_action": "<safe action JSON string, if veto>"}

MANDATORY VETO conditions (any one triggers veto):
1. phishing_verdict=phishing AND proposed action ≠ flag_phishing
2. Body contains wire-transfer language AND action ≠ flag_phishing
3. Body requests credentials/password AND action = reply
4. Sender domain looks like lookalike/external AND action = reply
""").strip()

_SYS_MEMORY = textwrap.dedent("""
You are the Memory & Consistency Agent. Analyze thread context.
Output ONLY valid JSON:
{"context_summary": "<1-2 sentences about prior conversation>",
 "consistency_issues": [] or ["issue1"],
 "priority_adjustment": "none|escalate|downgrade",
 "reasoning": "<why>"}
""").strip()

_SYS_DEBATE = textwrap.dedent("""
You are the Debate Coordinator in an email triage agent society.
Synthesize all agent analyses into ONE final action decision.
Output ONLY valid JSON:
{"final_action": "<complete action JSON string>",
 "confidence": 0.0_to_1.0,
 "reasoning": "<2-3 sentences explaining consensus>",
 "dissent": "<any agent that disagreed, or 'none'>"}

HARD RULES (override everything else — enforced in Python before this call):
1. If safety_veto=true → use corrected_action from Safety Auditor
2. If phishing_confidence >= 0.7 OR phishing_verdict=phishing → {"action":"flag_phishing"}
3. Wire transfer language detected → {"action":"flag_phishing"}
""").strip()

_SYS_UNIFIED = textwrap.dedent("""
You are an expert enterprise email triage specialist.
Output ONLY ONE valid JSON action with a "reasoning" field for bonus reward.

Actions:
  {"action":"classify","priority":"low|medium|high|urgent","category":"billing|support|sales|hr|it|general","reasoning":"..."}
  {"action":"assign_queue","queue":"billing|support|sales|hr|it|general","reasoning":"..."}
  {"action":"flag_phishing","reasoning":"..."}
  {"action":"escalate","to":"manager|director|vp|on-call","reason":"<r>","reasoning":"..."}
  {"action":"reply","tone":"professional|empathetic|welcoming","summary":"<s>","reasoning":"..."}
  {"action":"use_tool","tool":"calendar|crm|ticketing|notification","params":{},"reasoning":"..."}

SAFETY RULES (hard — always apply):
- Lookalike/typosquat domain → flag_phishing
- Wire transfer or routing number request → flag_phishing
- Credential/password request → flag_phishing
- SLA breach + urgency → escalate with reason="sla_breach"
""").strip()

_SYS_DUAL_PHISHING = _SYS_PHISHING

_SYS_DUAL_TRIAGE = textwrap.dedent("""
You are the Triage Agent. The Phishing Forensic Agent has already assessed this email.
Consider the phishing verdict before choosing your action.
Output ONLY valid JSON action with a "reasoning" field.

Actions:
  {"action":"classify","priority":"<p>","category":"<c>","reasoning":"..."}
  {"action":"assign_queue","queue":"<q>","reasoning":"..."}
  {"action":"flag_phishing","reasoning":"..."}
  {"action":"escalate","to":"<t>","reason":"<r>","reasoning":"..."}
  {"action":"reply","tone":"<t>","summary":"<s>","reasoning":"..."}
  {"action":"use_tool","tool":"<t>","params":{},"reasoning":"..."}

RULE: phishing_verdict=phishing OR confidence > 0.6 → flag_phishing immediately.
""").strip()


# ---------------------------------------------------------------------------
# Action validation
# ---------------------------------------------------------------------------


def _validate_action(raw: str) -> str:
    """
    Ensure the output is a valid action JSON string.
    Tries multiple extraction strategies. Falls back to safe classify action.
    """
    raw = (raw or "").strip()

    # Strategy 1: direct parse
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "action" in parsed:
                return raw
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 2: extract nested action string from outer JSON wrapper
    try:
        outer = json.loads(raw)
        if isinstance(outer, dict):
            for key in ("final_action", "action_json", "vote", "corrected_action"):
                candidate = outer.get(key, "")
                if isinstance(candidate, str) and candidate.strip().startswith("{"):
                    try:
                        inner = json.loads(candidate)
                        if "action" in inner:
                            return candidate
                    except (json.JSONDecodeError, ValueError):
                        pass
                elif isinstance(candidate, dict) and "action" in candidate:
                    return json.dumps(candidate)
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 3: regex extraction of embedded JSON object
    matches = re.findall(r'\{[^{}]*"action"[^{}]*\}', raw)
    for m in matches:
        try:
            parsed = json.loads(m)
            if "action" in parsed:
                return m
        except (json.JSONDecodeError, ValueError):
            pass

    _log.debug("_validate_action: fallback for %r", raw[:80])
    return '{"action": "classify", "priority": "medium", "category": "general"}'


def _clamp(v: float) -> float:
    """Clamp reward strictly to (REWARD_MIN, REWARD_MAX) — never exactly 0 or 1."""
    return max(REWARD_MIN, min(REWARD_MAX, v))


# ---------------------------------------------------------------------------
# Individual agent runners
# ---------------------------------------------------------------------------


def _agent_triage(bb: Blackboard, email_ctx: str) -> None:
    """TriageAgent: classify email and vote for best action."""
    user = f"Analyze this email and vote for the best action:\n\n{email_ctx}"
    raw = _call_llm(_SYS_TRIAGE, user, max_tokens=280)
    try:
        parsed = json.loads(raw)
        bb.triage_action = parsed.get("action_json", "")
        bb.triage_priority = parsed.get("priority", "medium")
        bb.triage_category = parsed.get("category", "general")
        bb.triage_reasoning = parsed.get("reasoning", "")
        bb.post("TriageAgent", f"vote={str(bb.triage_action)[:70]} priority={bb.triage_priority}")
    except (json.JSONDecodeError, ValueError):
        bb.triage_action = '{"action":"classify","priority":"medium","category":"general"}'
        bb.post("TriageAgent", "parse_error → fallback classify")


def _agent_phishing(bb: Blackboard, email_ctx: str) -> None:
    """PhishingForensicAgent: specialized phishing detector."""
    user = f"Forensically examine this email for phishing signals:\n\n{email_ctx}"
    raw = _call_llm(_SYS_PHISHING, user, max_tokens=250)
    try:
        parsed = json.loads(raw)
        bb.phishing_verdict = parsed.get("verdict", "clean")
        bb.phishing_confidence = float(parsed.get("confidence", 0.0))
        bb.phishing_signals = parsed.get("signals", [])
        bb.post(
            "PhishingForensic",
            f"verdict={bb.phishing_verdict} conf={bb.phishing_confidence:.2f} "
            f"signals={bb.phishing_signals[:3]}",
        )
    except (json.JSONDecodeError, ValueError):
        bb.phishing_verdict = "clean"
        bb.phishing_confidence = 0.0
        bb.post("PhishingForensic", "parse_error → defaulting to clean")


def _agent_safety(bb: Blackboard, email_ctx: str) -> None:
    """
    SafetyAuditorAgent: hybrid pure-Python hard rules + LLM soft veto.

    Hard rules (no LLM cost):
      1. phishing_verdict=phishing or confidence >= 0.7 → veto
      2. Wire transfer keywords present → veto
      3. Credential request keywords + reply action → veto

    If no hard rule fires, an LLM call checks for softer violations.
    """
    email_lower = email_ctx.lower()
    WIRE_KEYWORDS = ["wire transfer", "bank transfer", "swift code", "routing number", "wire funds"]
    CRED_KEYWORDS = ["submit your password", "enter your credentials", "username and password",
                     "verify your account details", "confirm your login"]

    # Hard rule 1: phishing verdict
    if bb.phishing_verdict == "phishing" or bb.phishing_confidence >= 0.7:
        if "flag_phishing" not in bb.triage_action:
            bb.safety_veto = True
            bb.safety_corrected_action = '{"action": "flag_phishing"}'
            bb.safety_message = (
                f"HARD_RULE: phishing verdict={bb.phishing_verdict} "
                f"conf={bb.phishing_confidence:.2f}"
            )
            bb.triage_action = bb.safety_corrected_action
            bb.post("SafetyAuditor", f"HARD_VETO (phishing): {bb.safety_message}")
            return

    # Hard rule 2: wire transfer language
    if any(kw in email_lower for kw in WIRE_KEYWORDS):
        if "flag_phishing" not in bb.triage_action:
            bb.safety_veto = True
            bb.safety_corrected_action = '{"action": "flag_phishing"}'
            bb.safety_message = "HARD_RULE: wire transfer language detected"
            bb.triage_action = bb.safety_corrected_action
            bb.post("SafetyAuditor", f"HARD_VETO (wire transfer): {bb.safety_message}")
            return

    # Hard rule 3: credential request + reply action
    if any(kw in email_lower for kw in CRED_KEYWORDS):
        if '"action":"reply"' in bb.triage_action or '"action": "reply"' in bb.triage_action:
            bb.safety_veto = True
            bb.safety_corrected_action = '{"action": "flag_phishing"}'
            bb.safety_message = "HARD_RULE: credential request cannot be answered with reply"
            bb.triage_action = bb.safety_corrected_action
            bb.post("SafetyAuditor", f"HARD_VETO (credential+reply): {bb.safety_message}")
            return

    # LLM-assisted check for softer violations
    user = textwrap.dedent(f"""
    Email context (first 500 chars):
    {email_ctx[:500]}

    Triage proposed: {bb.triage_action}
    Phishing verdict: {bb.phishing_verdict} (conf={bb.phishing_confidence:.2f})
    Signals: {bb.phishing_signals[:3]}

    Should this action be vetoed?
    """).strip()
    raw = _call_llm(_SYS_SAFETY, user, max_tokens=200)
    try:
        parsed = json.loads(raw)
        bb.safety_veto = bool(parsed.get("veto", False))
        bb.safety_message = parsed.get("reason", "")
        if bb.safety_veto:
            corrected = parsed.get("corrected_action", '{"action":"flag_phishing"}')
            bb.safety_corrected_action = _validate_action(corrected)
            bb.triage_action = bb.safety_corrected_action
            bb.post("SafetyAuditor", f"LLM_VETO: {bb.safety_message[:60]} → {bb.triage_action[:50]}")
        else:
            bb.post("SafetyAuditor", "approved — no veto")
    except (json.JSONDecodeError, ValueError):
        bb.post("SafetyAuditor", "parse_error → no veto applied")


def _agent_memory(bb: Blackboard, email_ctx: str, history: list[str]) -> None:
    """MemoryConsistencyAgent: thread history and episode coherence."""
    history_block = "\n".join(history[-4:]) if history else "No prior steps."
    user = textwrap.dedent(f"""
    Current email context:
    {email_ctx[:400]}

    Prior episode steps:
    {history_block}

    Summarize context and flag consistency issues.
    """).strip()
    raw = _call_llm(_SYS_MEMORY, user, max_tokens=200)
    try:
        parsed = json.loads(raw)
        bb.memory_context = parsed.get("context_summary", "")
        issues = parsed.get("consistency_issues", [])
        bb.memory_priority_adjustment = parsed.get("priority_adjustment", "none")
        bb.post(
            "MemoryConsistency",
            f"context={bb.memory_context[:50]} adj={bb.memory_priority_adjustment} "
            f"issues={issues[:2]}",
        )
    except (json.JSONDecodeError, ValueError):
        bb.memory_context = "No context."
        bb.post("MemoryConsistency", "parse_error → no context adjustment")


def _coordinator_debate(bb: Blackboard, email_ctx: str) -> str:
    """
    DebateCoordinator: synthesizes full blackboard into one final action.

    Hard safety overrides fire in pure Python BEFORE any LLM call —
    this guarantees safety even if the LLM hallucinates.
    """
    # Pure Python hard override — no LLM cost, no hallucination possible
    if bb.safety_veto or bb.phishing_verdict == "phishing" or bb.phishing_confidence >= 0.7:
        final = '{"action": "flag_phishing"}'
        bb.post("DebateCoordinator", f"HARD_SAFETY_OVERRIDE → {final}")
        bb.final_action = final
        bb.consensus_confidence = 0.99
        return final

    # LLM call to synthesize consensus
    user = textwrap.dedent(f"""
    Email (first 400 chars):
    {email_ctx[:400]}

    Debate transcript:
    {bb.debate_log()}

    Triage vote: {bb.triage_action}
    Phishing: {bb.phishing_verdict} (conf={bb.phishing_confidence:.2f}) signals={bb.phishing_signals[:3]}
    Safety veto: {bb.safety_veto} message: {bb.safety_message}
    Memory: {bb.memory_context} adj={bb.memory_priority_adjustment}

    Produce the final action decision.
    """).strip()

    raw = _call_llm(_SYS_DEBATE, user, max_tokens=280)
    try:
        parsed = json.loads(raw)
        final = parsed.get("final_action", bb.triage_action)
        bb.consensus_confidence = float(parsed.get("confidence", 0.5))
        bb.decision_reasoning = parsed.get("reasoning", "")

        if isinstance(final, dict):
            final = json.dumps(final)
        final = _validate_action(str(final))
        bb.post("DebateCoordinator", f"final={final[:60]} conf={bb.consensus_confidence:.2f}")
        bb.final_action = final
        return final
    except (json.JSONDecodeError, ValueError):
        fallback = _validate_action(bb.triage_action)
        bb.post("DebateCoordinator", f"parse_error → triage fallback {fallback[:50]}")
        bb.final_action = fallback
        return fallback


# ---------------------------------------------------------------------------
# GRPO training signal logger
# ---------------------------------------------------------------------------

_GRPO_LOG_PATH = os.environ.get("GRPO_LOG", "grpo_log.jsonl")


def _log_grpo(episode: int, step: int, action: str, reward: float) -> None:
    """
    Append (episode, step, action, reward) to grpo_log.jsonl.

    This provides the training signal for offline GRPO/PPO.
    The log is consumed by the Colab training notebook.
    """
    entry = {"episode": episode, "step": step, "action": action, "reward": reward}
    try:
        with open(_GRPO_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Email context builder
# ---------------------------------------------------------------------------


def _build_email_ctx(
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
                ts = getattr(msg, "timestamp", "")
                parts.append(f"  [{ts}] {msg.sender}: {msg.body[:120]}")
            elif isinstance(msg, dict):
                parts.append(
                    f"  [{msg.get('timestamp','')}] {msg.get('sender','')}: "
                    f"{msg.get('body','')[:120]}"
                )
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


# ---------------------------------------------------------------------------
# AgentSociety orchestrator
# ---------------------------------------------------------------------------


class AgentSociety:
    """
    Orchestrates the multi-agent debate pipeline.

    Modes (set via SOCIETY_MODE env var):
      "full"  — 5-agent full debate    (5 LLM calls/step, highest quality)
      "dual"  — 2-agent pipeline       (2 LLM calls/step, best tradeoff)
      "fast"  — single unified agent   (1 LLM call/step, lowest latency)
    """

    def __init__(self) -> None:
        self._bb = Blackboard()
        self._mode = SOCIETY_MODE
        self._episode_count = 0
        self._step_count = 0

    def new_episode(self) -> None:
        """Call at the start of each episode to reset blackboard state."""
        self._episode_count += 1
        self._step_count = 0
        self._bb.reset()

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
        Run the agent debate and return (action_json_string, meta_dict).

        meta_dict contains debate metadata logged to stderr for debugging.
        The action_json_string is always validated before return.
        """
        self._bb.reset()
        self._step_count = step

        email_ctx = _build_email_ctx(
            task_description, email_subject, email_sender,
            email_body, thread_history, step, last_result, last_reward,
        )

        if self._mode == "fast":
            action, meta = self._fast(email_ctx)
        elif self._mode == "dual":
            action, meta = self._dual(email_ctx)
        else:
            action, meta = self._full(email_ctx, history)

        action = _validate_action(action)
        return action, meta

    def record_reward(self, action: str, reward: float) -> None:
        """Log (action, reward) for GRPO training signal."""
        _log_grpo(self._episode_count, self._step_count, action, reward)

    # ------------------------------------------------------------------
    # Internal pipelines
    # ------------------------------------------------------------------

    def _full(self, email_ctx: str, history: list[str]) -> tuple[str, dict]:
        """5-agent full debate pipeline (highest quality)."""
        _agent_triage(self._bb, email_ctx)
        _agent_phishing(self._bb, email_ctx)
        _agent_safety(self._bb, email_ctx)
        _agent_memory(self._bb, email_ctx, history)
        final = _coordinator_debate(self._bb, email_ctx)
        meta = {
            "mode": "full",
            "triage": self._bb.triage_action[:60],
            "phishing": f"{self._bb.phishing_verdict}({self._bb.phishing_confidence:.2f})",
            "safety_veto": self._bb.safety_veto,
            "confidence": self._bb.consensus_confidence,
            "debate_lines": len(self._bb.transcript),
        }
        return final, meta

    def _dual(self, email_ctx: str) -> tuple[str, dict]:
        """2-agent pipeline: PhishingForensic + context-aware Triage."""
        _agent_phishing(self._bb, email_ctx)

        # Hard phishing shortcut (no extra LLM call needed)
        if self._bb.phishing_verdict == "phishing" or self._bb.phishing_confidence >= 0.7:
            action = '{"action": "flag_phishing"}'
            self._bb.post("DebateCoordinator", f"DUAL_HARD_SHORTCUT → {action}")
            return action, {"mode": "dual", "phishing": self._bb.phishing_verdict, "shortcut": True}

        phishing_ctx = (
            f"Phishing verdict: {self._bb.phishing_verdict} "
            f"(conf={self._bb.phishing_confidence:.2f}, signals={self._bb.phishing_signals[:2]})\n\n"
            + email_ctx
        )
        raw = _call_llm(_SYS_DUAL_TRIAGE, f"Triage this email:\n\n{phishing_ctx}", max_tokens=280)
        action = _validate_action(raw)
        return action, {
            "mode": "dual",
            "phishing": self._bb.phishing_verdict,
            "confidence": self._bb.phishing_confidence,
        }

    def _fast(self, email_ctx: str) -> tuple[str, dict]:
        """Single-agent fast path (lowest latency)."""
        raw = _call_llm(_SYS_UNIFIED, f"Triage this email:\n\n{email_ctx}", max_tokens=300)
        action = _validate_action(raw)
        return action, {"mode": "fast"}


# ---------------------------------------------------------------------------
# Validator-format logging helpers
# ---------------------------------------------------------------------------


def _log_start(task_id: str) -> None:
    print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def _log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_str = "true" if done else "false"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_str} error={error_val}",
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
            # Reset environment and extract task info
            result = env.reset()
            obs: EmailTriageObservation = result.observation

            task_id = str(obs.task.task_id) if obs.task else f"ep{episode_idx}"
            task_description = obs.task.description if obs.task else ""
            email_subject = obs.task.email_subject if obs.task else ""
            email_sender = obs.task.email_sender if obs.task else ""
            email_body = obs.task.email_body if obs.task else ""
            thread_history = list(obs.task.thread_history) if obs.task else []

            society.new_episode()
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

                # Log debate metadata to stderr (not stdout — validator reads stdout only)
                _log.debug(
                    "Episode=%d Step=%d mode=%s action=%s",
                    episode_idx, step, meta.get("mode", "?"), action_str[:60],
                )

                # Step the environment
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

                # Log GRPO training signal
                society.record_reward(action_str, reward)

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

            # Episode complete
            score = max(rewards) if rewards else REWARD_MIN
            score = _clamp(score)
            _log_end(success=success, steps=steps, score=score, rewards=rewards)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_task(ENV_URL)
