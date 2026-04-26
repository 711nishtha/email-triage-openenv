"""Data models for the Email Triage RL Environment."""

from __future__ import annotations

from enum import Enum
from typing import Any, NewType

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Core type aliases
# ---------------------------------------------------------------------------

TaskID = NewType("TaskID", int)
EpisodeID = NewType("EpisodeID", str)
StepCount = NewType("StepCount", int)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


# ---------------------------------------------------------------------------
# Task definition helpers
# ---------------------------------------------------------------------------


class StepCriteria(BaseModel):
    """A single required action step in a multi-step task."""

    action: str = Field(..., description="Required action type")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Required parameter subset — all keys must match (case-insensitive)",
    )


class StateCheck(BaseModel):
    """End-state assertion evaluated against tracker fields."""

    field: str = Field(..., description="EpisodeTracker attribute to inspect")
    expected: Any = Field(..., description="Expected value or substring")
    mode: str = Field(default="eq", description="'eq' | 'contains' | 'gte'")


class SuccessCriteria(BaseModel):
    """Machine-readable task completion criteria.

    Grading dispatch (in priority order):
      1. state_checks non-empty   → _grade_state_checks  (hard)
      2. steps length > 1         → _grade_multi_step     (medium)
      3. action_match set         → _grade_action_match   (easy)
    """

    action_match: StepCriteria | None = Field(
        default=None, description="Easy tier: single action check"
    )
    steps: list[StepCriteria] = Field(
        default_factory=list,
        description="Medium/Hard: ordered action sequence",
    )
    state_checks: list[StateCheck] = Field(
        default_factory=list,
        description="Hard tier: end-state assertions checked against tracker",
    )
    phishing_task: bool = Field(
        default=False,
        description="If True the agent MUST flag_phishing — any other action gives min reward",
    )


class EmailThread(BaseModel):
    """A single email message."""

    sender: str
    subject: str
    body: str
    timestamp: str = "2024-01-15 09:00"


class Task(BaseModel):
    """Full task definition (server-side only)."""

    task_id: TaskID
    difficulty: Difficulty
    description: str
    email: EmailThread
    thread_history: list[EmailThread] = Field(default_factory=list)
    success_criteria: SuccessCriteria
    hint_level_1: str = ""
    hint_level_2: str = ""
    hint_level_3: str = ""


class TaskInfo(BaseModel):
    """Agent-visible task info — success_criteria is hidden."""

    task_id: TaskID
    difficulty: Difficulty
    description: str
    email_subject: str
    email_sender: str
    email_body: str
    thread_history: list[EmailThread] = Field(default_factory=list)

    @classmethod
    def from_task(cls, task: Task) -> "TaskInfo":
        return cls(
            task_id=task.task_id,
            difficulty=task.difficulty,
            description=task.description,
            email_subject=task.email.subject,
            email_sender=task.email.sender,
            email_body=task.email.body,
            thread_history=task.thread_history,
        )


# ---------------------------------------------------------------------------
# Action & Observation
# ---------------------------------------------------------------------------


class EmailTriageAction(Action):
    """Agent action — a JSON string or the literal 'hint'."""

    action: str = Field(
        ...,
        description=(
            "JSON action string, e.g. "
            '\'{"action": "classify", "priority": "high", "category": "billing"}\''
        ),
    )


class EmailTriageObservation(Observation):
    """Observation returned after every step."""

    episode_id: EpisodeID
    step_count: StepCount
    task: TaskInfo | None = None
    last_action_result: str = ""
    last_action_valid: bool = True
    task_achieved: bool = False
    partial_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    hints_used: int = 0
    hint_text: str = ""
    done: bool = False
    reward: float = 0.0


# ---------------------------------------------------------------------------
# Environment state
# ---------------------------------------------------------------------------


class TrackerState(BaseModel):
    """Serialisable snapshot of EpisodeTracker."""

    step_count: int = 0
    hints_used: int = 0
    progress: float = 0.0
    actions_taken: list[str] = Field(default_factory=list)
    credited_steps: list[str] = Field(default_factory=list)
    phishing_missed: bool = False
    tool_calls: list[str] = Field(default_factory=list)
    flags_raised: list[str] = Field(default_factory=list)


class EmailTriageState(State):
    """Full environment state (server-side)."""

    current_task: Task | None = None
    tracker: TrackerState = Field(default_factory=TrackerState)
    episode_id: str = ""
    step_count: int = 0
    current_tier: str = "easy"
