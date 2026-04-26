"""Per-episode state tracker for the Email Triage RL Environment."""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class StepRecord(BaseModel):
    """A single agent action recorded within an episode."""

    raw_action: str
    action_type: str
    action_dict: dict = Field(default_factory=dict)
    valid: bool = True
    step_number: int = Field(ge=0)


class EpisodeTracker:
    """Tracks action history and derived state within a single episode.

    Mirrors the reference EpisodeTracker API so task_grader.py can call
    the same interface patterns.
    """

    def __init__(self) -> None:
        self._history: list[StepRecord] = []
        self._step_counter: int = 0
        self._previous_progress: float = 0.0
        self._credited_steps: list[str] = []
        self._hints_used: int = 0
        self.phishing_missed: bool = False
        self.tool_calls: list[str] = []
        self.flags_raised: list[str] = []
        self.actions_taken: list[str] = []

    def reset(self) -> None:
        self._history.clear()
        self._step_counter = 0
        self._previous_progress = 0.0
        self._credited_steps.clear()
        self._hints_used = 0
        self.phishing_missed = False
        self.tool_calls.clear()
        self.flags_raised.clear()
        self.actions_taken.clear()

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(self, raw_action: str, action_dict: dict, valid: bool) -> StepRecord:
        action_type = action_dict.get("action", "unknown")
        record = StepRecord(
            raw_action=raw_action,
            action_type=action_type,
            action_dict=action_dict,
            valid=valid,
            step_number=self._step_counter,
        )
        self._history.append(record)
        self._step_counter += 1

        # Side-effects
        self.actions_taken.append(action_type)
        if action_type == "use_tool":
            tool = action_dict.get("tool") or action_dict.get("params", {}).get("tool", "")
            if tool:
                self.tool_calls.append(tool)
        if action_type == "flag_phishing":
            self._raise_flag("phishing")
        if action_type == "escalate":
            reason = str(action_dict.get("reason", "")).lower()
            if "sla" in reason or "breach" in reason:
                self._raise_flag("sla_breach")

        return record

    def record_hint(self) -> int:
        """Record a hint request. Returns the new hint count (1-indexed)."""
        self._hints_used += 1
        return self._hints_used

    # ------------------------------------------------------------------
    # Query helpers (used by task_grader)
    # ------------------------------------------------------------------

    def has_executed_action(self, action_type: str, params: dict | None = None) -> bool:
        """Check if a successful action matching (type, params subset) exists."""
        for record in self._history:
            if not record.valid:
                continue
            if record.action_type != action_type:
                continue
            if params:
                if not _params_match(record.action_dict, params):
                    continue
            return True
        return False

    def is_step_credited(self, step_key: str) -> bool:
        return step_key in self._credited_steps

    def credit_step(self, step_key: str) -> bool:
        """Mark step as credited (deduplication). Returns True if newly credited."""
        if step_key in self._credited_steps:
            return False
        self._credited_steps.append(step_key)
        return True

    def get_credited_count(self) -> int:
        return len(self._credited_steps)

    def _raise_flag(self, flag: str) -> None:
        if flag not in self.flags_raised:
            self.flags_raised.append(flag)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def step_count(self) -> int:
        return self._step_counter

    @property
    def hints_used(self) -> int:
        return self._hints_used

    @property
    def previous_progress(self) -> float:
        return self._previous_progress

    @previous_progress.setter
    def previous_progress(self, value: float) -> None:
        self._previous_progress = value

    @property
    def command_history(self) -> list[StepRecord]:
        return list(self._history)


def _params_match(action_dict: dict, required: dict) -> bool:
    for k, v in required.items():
        actual = action_dict.get(k)
        if actual is None:
            actual = action_dict.get("params", {}).get(k)
        if actual is None:
            return False
        if str(actual).lower() != str(v).lower():
            return False
    return True
