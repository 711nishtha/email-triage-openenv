"""
models.py — Pydantic models for OpenEnv API compliance.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


# ─────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────

class ActionType(str, Enum):
    triage = "triage"
    escalate = "escalate"
    use_tool = "use_tool"
    done = "done"


class Priority(str, Enum):
    urgent = "urgent"
    high = "high"
    medium = "medium"
    low = "low"


class Category(str, Enum):
    phishing = "phishing"
    urgent_business = "urgent_business"
    internal_task = "internal_task"
    marketing = "marketing"
    hr = "hr"
    legal = "legal"
    it_support = "it_support"
    finance = "finance"
    spam = "spam"


class RouteTarget(str, Enum):
    security = "security"
    executive = "executive"
    manager = "manager"
    it = "it"
    hr = "hr"
    finance = "finance"
    archive = "archive"
    trash = "trash"


# ─────────────────────────────────────────────
# Email Model
# ─────────────────────────────────────────────

class Email(BaseModel):
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    # Hidden fields (not exposed in observation, used by grader)
    _expected_priority: Optional[str] = None
    _expected_category: Optional[str] = None
    _expected_route: Optional[str] = None

    def to_observation(self) -> Dict[str, Any]:
        """Return only fields visible to the agent."""
        return {
            "id": self.id,
            "sender": self.sender,
            "subject": self.subject,
            "body": self.body,
            "timestamp": self.timestamp,
        }


class EmailWithGroundTruth(BaseModel):
    """Full email including grading labels (never sent to agent)."""
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    expected_priority: str
    expected_category: str
    expected_route: str

    def to_observation(self) -> Dict[str, Any]:
        """Return only fields visible to the agent (no ground truth labels)."""
        return {
            "id": self.id,
            "sender": self.sender,
            "subject": self.subject,
            "body": self.body,
            "timestamp": self.timestamp,
        }


# ─────────────────────────────────────────────
# Action Model
# ─────────────────────────────────────────────

class Action(BaseModel):
    action_type: ActionType
    email_id: Optional[str] = None
    priority: Optional[Priority] = None
    category: Optional[Category] = None
    route_to: Optional[RouteTarget] = None
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# Observation Model
# ─────────────────────────────────────────────

class Observation(BaseModel):
    inbox: List[Dict[str, Any]]
    triaged: List[Dict[str, Any]] = Field(default_factory=list)
    step_count: int = 0
    task_id: str = ""
    max_steps: int = 20
    available_tools: List[str] = Field(default_factory=list)
    message: str = ""


# ─────────────────────────────────────────────
# Step Response Model
# ─────────────────────────────────────────────

class StepResponse(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────
# State Model
# ─────────────────────────────────────────────

class EnvironmentState(BaseModel):
    task_id: str
    step_count: int
    max_steps: int
    inbox: List[Dict[str, Any]]
    triaged: List[Dict[str, Any]]
    cumulative_reward: float
    done: bool
    available_tools: List[str]
    tool_results: List[Dict[str, Any]]


# ─────────────────────────────────────────────
# Reset Request
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    task_id: Optional[str] = "easy"


# ─────────────────────────────────────────────
# Tool Result
# ─────────────────────────────────────────────

class ToolResult(BaseModel):
    tool_name: str
    success: bool
    output: Any
    error: Optional[str] = None
