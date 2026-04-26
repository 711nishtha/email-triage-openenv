"""
agents/society.py — Email Triage Agent Society (modular re-export)
===================================================================
This module re-exports the AgentSociety class from inference.py so that
external scripts (e.g., evaluation harnesses, training scripts) can import
the society without running inference.

For the full implementation, see inference.py.
The split is intentional:
  - inference.py is the self-contained executable (OpenEnv validator runs it)
  - agents/society.py is the importable module (training scripts, tests use it)

Usage:
    from agents.society import AgentSociety
    society = AgentSociety()
    action, meta = society.deliberate(...)
"""

from __future__ import annotations

# Re-export everything from inference.py
# This allows `from agents.society import AgentSociety, Blackboard, _validate_action`
from inference import (
    AgentSociety,
    Blackboard,
    _validate_action,
    _build_email_ctx,
    _agent_triage,
    _agent_phishing,
    _agent_safety,
    _agent_memory,
    _coordinator_debate,
    _call_llm,
    _clamp,
    _log_grpo,
    SOCIETY_MODE,
    MODEL_NAME,
    API_BASE_URL,
)

__all__ = [
    "AgentSociety",
    "Blackboard",
    "_validate_action",
    "_build_email_ctx",
    "_agent_triage",
    "_agent_phishing",
    "_agent_safety",
    "_agent_memory",
    "_coordinator_debate",
    "_call_llm",
    "_clamp",
    "_log_grpo",
    "SOCIETY_MODE",
    "MODEL_NAME",
    "API_BASE_URL",
]
