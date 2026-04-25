"""
agents/society.py — Email Triage Agent Society
================================================
Implements the multi-agent colony used by inference.py.

Colony members:
  1. TriageAgent          — primary email classifier and action voter
  2. PhishingForensicAgent — specialized phishing/social-engineering detector
  3. SafetyAuditorAgent   — symbolic + LLM veto layer
  4. MemoryConsistencyAgent — thread history and episode coherence tracker
  5. DebateCoordinator    — synthesizes all votes into one final action

Communication pattern:
  Shared Blackboard → each agent reads prior posts → posts its analysis →
  Debate Coordinator reads full transcript → outputs final action JSON

GRPO-style learning signal:
  The society logs per-step (action, reward) pairs to grpo_log.jsonl.
  A separate training script can consume this for offline GRPO/PPO.
  The society itself does NOT do online learning — it is pure inference.

Performance modes:
  SOCIETY_MODE="full"  — 5 LLM calls per step (full debate)
  SOCIETY_MODE="fast"  — 1 LLM call per step (single unified agent)
  SOCIETY_MODE="dual"  — 2 LLM calls: PhishingForensic + unified triage
                         Best latency/quality tradeoff for proxy rate limits.

All outputs:
  - return valid JSON action strings that the environment can parse
  - are verified by _validate_action() before returning
  - Hard safety overrides (phishing/wire-transfer) are applied in pure Python
    before any LLM call to the Debate Coordinator

Zero paid API calls. All LLM calls go through API_BASE_URL + API_KEY.
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

GRPO_LOG = Path(os.getenv("GRPO_LOG", "grpo_log.jsonl"))
SOCIETY_MODE = os.getenv("SOCIETY_MODE", "full")

# ---------------------------------------------------------------------------
# Blackboard
# ---------------------------------------------------------------------------


@dataclass
class Blackboard:
    """
    Shared memory for the agent colony.
    Agents post structured analyses; the Coordinator reads everything.
    """
    # Triage Agent output
    triage_action: str = ""
    triage_priority: str = "medium"
    triage_category: str = "general"
    triage_reasoning: str = ""

    # Phishing Forensic output
    phishing_verdict: str = "clean"          # "phishing" | "suspicious" | "clean"
    phishing_confidence: float = 0.0
    phishing_signals: list[str] = field(default_factory=list)

    # Safety Auditor output
    safety_veto: bool = False
    safety_corrected_action: str = ""
    safety_message: str = ""

    # Memory/Consistency output
    memory_context: str = ""
    memory_priority_adjustment: str = "none"  # "none" | "escalate" | "downgrade"
    memory_issues: list[str] = field(default_factory=list)

    # Debate transcript
    transcript: list[str] = field(default_factory=list)

    # Final output
    final_action: str = ""
    consensus_confidence: float = 0.5
    decision_reasoning: str = ""

    def post(self, agent: str, msg: str) -> None:
        self.transcript.append(f"[{agent}] {msg}")

    def debate_log(self, max_lines: int = 10) -> str:
        return "\n".join(self.transcript[-max_lines:])

    def reset(self) -> None:
        self.__init__()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# System prompts (concise — minimize token cost)
# ---------------------------------------------------------------------------

_SYS_TRIAGE = textwrap.dedent("""
You are the Triage Agent. Classify the email and choose the best action.
Output ONLY JSON:
{"action_json": "<action string>", "priority": "low|medium|high|urgent",
 "category": "billing|support|sales|hr|it|general",
 "reasoning": "<2 sentences max>"}

Action strings (choose ONE):
  {"action":"classify","priority":"<p>","category":"<c>"}
  {"action":"assign_queue","queue":"<billing|support|sales|hr|it|general>"}
  {"action":"escalate","to":"<manager|director|vp|on-call>","reason":"<r>"}
  {"action":"reply","tone":"<professional|empathetic|welcoming>","summary":"<s>"}
  {"action":"use_tool","tool":"<calendar|crm|ticketing|notification>","params":{}}
  {"action":"flag_phishing"}

Rules: ONE action. If ANY phishing signals visible → vote flag_phishing.
""").strip()

_SYS_PHISHING = textwrap.dedent("""
You are the Phishing Forensic Agent. Analyze ONLY for phishing/fraud.
Output ONLY JSON:
{"verdict": "phishing|suspicious|clean",
 "confidence": 0.00_to_1.00,
 "signals": ["signal1","signal2"],
 "recommendation": "flag_phishing|proceed|verify"}

Phishing signals to check:
- Lookalike/typosquat domain (paypa1, amaz0n, microsofft, -secure-msg., .ru, .biz)
- Wire transfer or bank routing requests
- Credential/password requests
- Urgency manipulation ("act now or lose access")
- CEO/executive impersonation from external domain
- Mismatched sender name vs domain
""").strip()

_SYS_SAFETY = textwrap.dedent("""
You are the Safety Auditor. Decide if the proposed action must be vetoed.
Output ONLY JSON:
{"veto": true|false,
 "reason": "<why, if veto>",
 "corrected_action": "<safe action JSON string, if veto>"}

MANDATORY VETO conditions:
1. phishing_verdict=phishing AND proposed action ≠ flag_phishing
2. Body contains wire-transfer language AND action ≠ flag_phishing
3. Body requests credentials AND action = reply
4. Sender domain is lookalike/suspicious AND action = reply
""").strip()

_SYS_MEMORY = textwrap.dedent("""
You are the Memory & Consistency Agent. Analyze thread context.
Output ONLY JSON:
{"context_summary": "<1-2 sentences>",
 "consistency_issues": [] or ["issue"],
 "priority_adjustment": "none|escalate|downgrade",
 "reasoning": "<why>"}
""").strip()

_SYS_DEBATE = textwrap.dedent("""
You are the Debate Coordinator. Synthesize the agent debate into ONE final action.
Output ONLY JSON:
{"final_action": "<complete action JSON string>",
 "confidence": 0.0_to_1.0,
 "reasoning": "<2-3 sentences>",
 "dissent": "<any agent that disagreed, or none>"}

HARD RULES (override everything else):
1. If safety_veto=true → use corrected_action from Safety Auditor
2. If phishing_confidence >= 0.7 OR phishing_verdict=phishing → output {"action":"flag_phishing"}
3. Wire transfer language in email → output {"action":"flag_phishing"}
""").strip()

_SYS_UNIFIED = textwrap.dedent("""
You are an expert enterprise email triage specialist.
Output ONLY valid JSON action. Include "reasoning" field for bonus reward.

Actions:
  {"action":"classify","priority":"low|medium|high|urgent","category":"billing|support|sales|hr|it|general","reasoning":"..."}
  {"action":"assign_queue","queue":"billing|support|sales|hr|it|general","reasoning":"..."}
  {"action":"flag_phishing","reasoning":"..."}
  {"action":"escalate","to":"manager|director|vp|on-call","reason":"<r>","reasoning":"..."}
  {"action":"reply","tone":"professional|empathetic|welcoming","summary":"<s>","reasoning":"..."}
  {"action":"use_tool","tool":"calendar|crm|ticketing|notification","params":{},"reasoning":"..."}

SAFETY (hard rules — always apply):
- Lookalike/typosquat domain → flag_phishing
- Wire transfer request → flag_phishing
- Credential/password request → flag_phishing
- SLA breach language + urgency → escalate with reason="sla_breach"
""").strip()

_SYS_DUAL_PHISHING = _SYS_PHISHING  # same prompt, used in dual mode
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

RULE: If phishing verdict is phishing or confidence > 0.6 → flag_phishing.
""").strip()


# ---------------------------------------------------------------------------
# LLM call wrapper (imported at use time to avoid circular imports)
# ---------------------------------------------------------------------------

def _llm_call(system: str, user: str, max_tokens: int = 300) -> str:
    """
    Call the LLM configured in inference.py via _call_llm.
    Falls back gracefully.
    """
    try:
        # Import from inference.py runtime context
        from inference import _call_llm
        return _call_llm(system, user, max_tokens=max_tokens)
    except ImportError:
        # Standalone usage — read config from env
        import os
        from openai import OpenAI
        client = OpenAI(
            base_url=os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1"),
            api_key=os.environ.get("API_KEY", os.environ.get("HF_TOKEN", "placeholder")),
        )
        try:
            resp = client.chat.completions.create(
                model=os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"),
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_tokens=max_tokens,
                temperature=0.15,
            )
            text = (resp.choices[0].message.content or "").strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(l for l in lines if not l.startswith("```")).strip()
            return text
        except Exception as exc:
            logger.warning("LLM call failed: %s", exc)
            return ""


# ---------------------------------------------------------------------------
# Action validation
# ---------------------------------------------------------------------------

def _validate_action(raw: str) -> str:
    """
    Ensure the output is a valid action JSON string.
    Tries multiple extraction strategies before returning a safe fallback.
    """
    raw = (raw or "").strip()

    # Direct parse
    if raw.startswith("{"):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "action" in parsed:
                return raw
        except (json.JSONDecodeError, ValueError):
            pass

    # Nested string (Debate Coordinator sometimes returns the action as a JSON string value)
    try:
        outer = json.loads(raw)
        if isinstance(outer, dict):
            # Look for final_action or action_json fields
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

    # Regex extraction of embedded JSON object
    matches = re.findall(r'\{[^{}]*"action"[^{}]*\}', raw)
    for m in matches:
        try:
            parsed = json.loads(m)
            if "action" in parsed:
                return m
        except (json.JSONDecodeError, ValueError):
            pass

    # Safe fallback
    logger.debug("_validate_action: could not parse '%s', using fallback", raw[:80])
    return '{"action": "classify", "priority": "medium", "category": "general"}'


# ---------------------------------------------------------------------------
# Individual agent runners
# ---------------------------------------------------------------------------

def _agent_triage(bb: Blackboard, email_ctx: str) -> None:
    user = f"Analyze this email and vote for the best action:\n\n{email_ctx}"
    raw = _llm_call(_SYS_TRIAGE, user, max_tokens=280)
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
    user = f"Forensically examine this email for phishing signals:\n\n{email_ctx}"
    raw = _llm_call(_SYS_PHISHING, user, max_tokens=250)
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
    """Safety Auditor: LLM-assisted veto + pure Python hard rules."""

    # Pure Python hard rules (free, instant, reliable)
    email_lower = email_ctx.lower()
    wire_keywords = ["wire transfer", "bank transfer", "swift code", "routing number", "wire funds"]
    cred_keywords = ["submit your password", "enter your credentials", "username and password",
                     "verify your account details", "confirm your login"]

    hard_veto = False
    hard_reason = ""

    if bb.phishing_verdict == "phishing" or bb.phishing_confidence >= 0.7:
        if '"action":"flag_phishing"' not in bb.triage_action and '"action": "flag_phishing"' not in bb.triage_action:
            hard_veto = True
            hard_reason = f"Phishing verdict={bb.phishing_verdict} conf={bb.phishing_confidence:.2f}"

    if any(kw in email_lower for kw in wire_keywords):
        if "flag_phishing" not in bb.triage_action:
            hard_veto = True
            hard_reason = "Wire transfer language detected"

    if hard_veto:
        bb.safety_veto = True
        bb.safety_corrected_action = '{"action": "flag_phishing"}'
        bb.safety_message = f"HARD_RULE: {hard_reason}"
        bb.post("SafetyAuditor", f"HARD_VETO: {hard_reason}")
        return

    # LLM-assisted check for softer violations
    user = textwrap.dedent(f"""
    Email context (abbreviated):
    {email_ctx[:500]}

    Triage proposed: {bb.triage_action}
    Phishing verdict: {bb.phishing_verdict} (conf={bb.phishing_confidence:.2f})

    Should this action be vetoed?
    """).strip()
    raw = _llm_call(_SYS_SAFETY, user, max_tokens=200)
    try:
        parsed = json.loads(raw)
        bb.safety_veto = bool(parsed.get("veto", False))
        bb.safety_message = parsed.get("reason", "")
        if bb.safety_veto:
            corrected = parsed.get("corrected_action", '{"action":"flag_phishing"}')
            bb.safety_corrected_action = corrected
            bb.triage_action = _validate_action(corrected)  # override
            bb.post("SafetyAuditor", f"LLM_VETO: {bb.safety_message[:60]} → {bb.triage_action[:50]}")
        else:
            bb.post("SafetyAuditor", "approved")
    except (json.JSONDecodeError, ValueError):
        bb.post("SafetyAuditor", "parse_error → no veto")


def _agent_memory(bb: Blackboard, email_ctx: str, history: list[str]) -> None:
    history_block = "\n".join(history[-4:]) if history else "No prior steps."
    user = textwrap.dedent(f"""
    Current email context:
    {email_ctx[:400]}

    Prior episode steps:
    {history_block}
    """).strip()
    raw = _llm_call(_SYS_MEMORY, user, max_tokens=200)
    try:
        parsed = json.loads(raw)
        bb.memory_context = parsed.get("context_summary", "")
        bb.memory_issues = parsed.get("consistency_issues", [])
        bb.memory_priority_adjustment = parsed.get("priority_adjustment", "none")
        bb.post(
            "MemoryConsistency",
            f"context={bb.memory_context[:50]} adj={bb.memory_priority_adjustment} "
            f"issues={bb.memory_issues[:2]}",
        )
    except (json.JSONDecodeError, ValueError):
        bb.memory_context = "No context."
        bb.post("MemoryConsistency", "parse_error → no context")


def _coordinator_debate(bb: Blackboard, email_ctx: str) -> str:
    """
    Debate Coordinator: reads the full blackboard and produces final action.
    Hard safety overrides are applied in pure Python first (no LLM cost).
    """
    # Hard override: phishing/veto → immediate flag_phishing, no LLM needed
    if bb.safety_veto or bb.phishing_verdict == "phishing" or bb.phishing_confidence >= 0.7:
        final = '{"action": "flag_phishing"}'
        bb.post("DebateCoordinator", f"HARD_SAFETY → {final}")
        bb.final_action = final
        bb.consensus_confidence = 0.99
        return final

    user = textwrap.dedent(f"""
    Email (brief):
    {email_ctx[:400]}

    Debate transcript:
    {bb.debate_log()}

    Triage vote: {bb.triage_action}
    Phishing: {bb.phishing_verdict} (conf={bb.phishing_confidence:.2f}) signals={bb.phishing_signals[:3]}
    Safety veto: {bb.safety_veto} message: {bb.safety_message}
    Memory: {bb.memory_context} adj={bb.memory_priority_adjustment}

    Produce the final action.
    """).strip()

    raw = _llm_call(_SYS_DEBATE, user, max_tokens=280)
    try:
        parsed = json.loads(raw)
        final = parsed.get("final_action", bb.triage_action)
        bb.consensus_confidence = float(parsed.get("confidence", 0.5))
        bb.decision_reasoning = parsed.get("reasoning", "")

        if isinstance(final, dict):
            final = json.dumps(final)
        final = _validate_action(str(final))
        bb.post(
            "DebateCoordinator",
            f"final={final[:60]} conf={bb.consensus_confidence:.2f}",
        )
        bb.final_action = final
        return final
    except (json.JSONDecodeError, ValueError):
        fallback = _validate_action(bb.triage_action)
        bb.post("DebateCoordinator", f"parse_error → triage fallback {fallback[:50]}")
        bb.final_action = fallback
        return fallback


# ---------------------------------------------------------------------------
# GRPO signal logger
# ---------------------------------------------------------------------------

def _log_grpo(episode: int, step: int, action: str, reward: float) -> None:
    """
    Append (episode, step, action, reward) to grpo_log.jsonl.
    A separate offline trainer can use this for GRPO/PPO updates.
    """
    entry = {
        "episode": episode,
        "step": step,
        "action": action,
        "reward": reward,
    }
    try:
        with open(GRPO_LOG, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError:
        pass


# ---------------------------------------------------------------------------
# AgentSociety orchestrator (used by inference.py)
# ---------------------------------------------------------------------------

class AgentSociety:
    """
    Public interface for inference.py.
    Manages one Blackboard per episode and dispatches agent calls.
    """

    def __init__(self) -> None:
        self._bb = Blackboard()
        self._mode = SOCIETY_MODE
        self._episode_count = 0
        self._step_count = 0

    def new_episode(self) -> None:
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
        Orchestrate the agent debate and return (action_json, meta_dict).

        meta_dict contains debate metadata for logging/dashboard.
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
        """5-agent full debate pipeline."""
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
        """2-agent pipeline: PhishingForensic + Triage with context."""
        _agent_phishing(self._bb, email_ctx)

        # Hard phishing shortcut
        if self._bb.phishing_verdict == "phishing" or self._bb.phishing_confidence >= 0.7:
            action = '{"action": "flag_phishing"}'
            return action, {"mode": "dual", "phishing": self._bb.phishing_verdict, "shortcut": True}

        phishing_ctx = (
            f"Phishing verdict: {self._bb.phishing_verdict} "
            f"(conf={self._bb.phishing_confidence:.2f}, signals={self._bb.phishing_signals[:2]})\n\n"
            + email_ctx
        )
        raw = _llm_call(_SYS_DUAL_TRIAGE, f"Triage this email:\n\n{phishing_ctx}", max_tokens=280)
        action = _validate_action(raw)
        return action, {"mode": "dual", "phishing": self._bb.phishing_verdict, "confidence": self._bb.phishing_confidence}

    def _fast(self, email_ctx: str) -> tuple[str, dict]:
        """Single-agent fast path."""
        raw = _llm_call(_SYS_UNIFIED, f"Triage this email:\n\n{email_ctx}", max_tokens=300)
        action = _validate_action(raw)
        return action, {"mode": "fast"}


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
                    f"  [{msg.get('timestamp','')}] {msg.get('sender','')}: {msg.get('body','')[:120]}"
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
