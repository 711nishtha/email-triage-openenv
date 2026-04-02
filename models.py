"""
models.py — Pydantic typed models for the Advanced Enterprise Email Triage OpenEnv.

All data flowing through reset(), step(), and state() is fully typed here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


# ── Email primitives ──────────────────────────────────────────────────────────

class EmailMessage(BaseModel):
    """A single email in the inbox or thread history."""

    email_id: str = Field(..., description="Unique identifier for this email")
    thread_id: Optional[str] = Field(None, description="Thread this email belongs to, if any")
    subject: str
    sender: str = Field(..., description="Full sender email address")
    sender_display_name: str = Field(..., description="Human-readable sender name")
    recipients: List[str] = Field(default_factory=list)
    cc: List[str] = Field(default_factory=list)
    timestamp: datetime
    body: str = Field(..., description="Email body text (plain text)")
    has_attachments: bool = False
    attachment_names: List[str] = Field(default_factory=list)
    links: List[str] = Field(default_factory=list, description="URLs found in email body")
    is_reply: bool = False
    reply_to: Optional[str] = Field(None, description="email_id this is a reply to")

    # Ground-truth labels (hidden from agent, used by grader)
    # Using private-prefixed fields to indicate they are not for the agent
    true_priority: Optional[str] = Field(None, alias="_true_priority")
    true_category: Optional[str] = Field(None, alias="_true_category")
    true_route: Optional[str] = Field(None, alias="_true_route")
    is_phishing: Optional[bool] = Field(None, alias="_is_phishing")
    is_bec: Optional[bool] = Field(None, alias="_is_bec")  # Business Email Compromise

    class Config:
        populate_by_name = True


class SenderProfile(BaseModel):
    """Metadata about a known sender — used for dynamic importance scoring."""

    email: str
    display_name: str
    domain: str
    is_internal: bool = False
    is_vip: bool = False                     # C-suite, board, key partners
    is_known_vendor: bool = False
    reputation_score: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Domain/sender reputation 0-1 (phishing domains score low)"
    )
    previous_interactions: int = 0          # Count of prior legitimate emails
    department: Optional[str] = None
    job_title: Optional[str] = None
    is_flagged_suspicious: bool = False     # Pre-flagged by security tooling


# ── Action models ────────────────────────────────────────────────────────────

PriorityLevel = Literal["critical", "high", "medium", "low", "spam"]

EmailCategory = Literal[
    "security_incident",
    "executive_request",
    "hr_matter",
    "vendor_contract",
    "it_support",
    "team_update",
    "customer_escalation",
    "phishing",
    "newsletter",
    "other",
]

RoutingTarget = Literal[
    "security_team",
    "ceo_office",
    "cfo_office",
    "hr_team",
    "legal_team",
    "it_helpdesk",
    "engineering_team",
    "sales_team",
    "finance_team",
    "operations_team",
    "customer_success",
    "executive_assistant",
    "archive",
    "spam_folder",
]


class TriageAction(BaseModel):
    """
    The structured action an agent submits each step.

    action_type determines which fields are relevant:
    - "triage"    → email_id, priority, category, route_to, reasoning
    - "use_tool"  → tool_name, tool_params
    - "escalate"  → email_id, escalation_target, escalation_reason
    - "done"      → no additional fields
    """

    action_type: Literal["triage", "use_tool", "escalate", "done"] = Field(
        ..., description="Type of action to perform"
    )

    # ── Triage fields ──
    email_id: Optional[str] = Field(None, description="ID of email being triaged")
    priority: Optional[PriorityLevel] = None
    category: Optional[EmailCategory] = None
    route_to: Optional[RoutingTarget] = None
    reasoning: Optional[str] = Field(
        None,
        description="Agent's chain-of-thought for this triage decision (logged, not scored)",
    )

    # ── Tool call fields ──
    tool_name: Optional[Literal["calendar_check", "kb_search", "sender_lookup"]] = None
    tool_params: Optional[Dict[str, Any]] = Field(default=None)

    # ── Escalation fields ──
    escalation_target: Optional[Literal["security_team", "ceo_office", "legal_team"]] = None
    escalation_reason: Optional[str] = None


# ── Observation models ───────────────────────────────────────────────────────

class EmailObservation(BaseModel):
    """
    Full observation returned to the agent after reset() or step().
    """

    task_id: str
    step: int = Field(..., ge=0)
    max_steps: int = Field(..., ge=1)

    inbox: List[EmailMessage] = Field(
        ..., description="Emails pending triage in this episode"
    )
    thread_history: Dict[str, List[EmailMessage]] = Field(
        default_factory=dict,
        description="Map of thread_id → list of prior emails (context only, already triaged)",
    )
    sender_profiles: Dict[str, SenderProfile] = Field(
        default_factory=dict,
        description="Map of sender address → SenderProfile metadata",
    )
    available_tools: List[str] = Field(
        default_factory=list,
        description="Tool names the agent may call this step",
    )
    current_score: float = Field(
        default=0.0, ge=-1.0, le=1.0,
        description="Cumulative normalised score so far"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="Security/compliance warnings triggered so far",
    )
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class StepResponse(BaseModel):
    """Response returned by /step — mirrors gym-style (obs, reward, done, info)."""

    observation: EmailObservation
    reward: float = Field(..., ge=-1.0, le=1.0)
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ── State model ──────────────────────────────────────────────────────────────

class EnvironmentState(BaseModel):
    """
    Full internal state — returned by /state for debugging or evaluation.
    Includes ground-truth labels that are hidden from the agent during the episode.
    """

    task_id: str
    episode_id: str
    step: int
    max_steps: int
    done: bool
    total_reward: float
    emails: List[EmailMessage]
    agent_decisions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Log of all agent triage/escalation decisions this episode",
    )
    tool_call_log: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Log of all tool calls this episode",
    )
    pending_email_ids: List[str] = Field(
        default_factory=list, description="Email IDs not yet triaged"
    )
    triaged_email_ids: List[str] = Field(
        default_factory=list, description="Email IDs already triaged"
    )
    warnings: List[str] = Field(default_factory=list)
    grader_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-component score breakdown from grader",
    )


# ── Request models ───────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    """Request body for POST /reset."""

    task_id: Literal["easy", "medium", "hard"] = "easy"
    seed: Optional[int] = Field(
        None, description="Optional random seed for reproducible episodes"
    )
