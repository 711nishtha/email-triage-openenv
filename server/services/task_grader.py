"""
Task grading engine for the Email Triage RL Environment.

Mirrors the reference TaskGrader structure:
  - GradeResult is a Pydantic BaseModel
  - Dispatch based on which SuccessCriteria fields are populated
  - _compute_reward applies progressive shaping
  - Rewards clamped to [0.0, 0.99] server-side (inference.py adds the 0.001 floor)

Dispatch priority:
  1. state_checks present          → _grade_state_checks   (hard)
  2. steps length > 1              → _grade_multi_step      (medium)
  3. action_match set              → _grade_action_match    (easy)
"""

from __future__ import annotations

import json
import logging

from pydantic import BaseModel, Field

from models import StateCheck, StepCriteria, SuccessCriteria, Task
from server.services.episode_tracker import EpisodeTracker, StepRecord, _params_match

logger = logging.getLogger(__name__)

HINT_DECAY = 0.85


class GradeResult(BaseModel):
    """Outcome of grading a single step."""

    task_achieved: bool = False
    partial_progress: float = Field(default=0.0, ge=0.0, le=1.0)
    reward: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""
    phishing_violation: bool = False


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------


def _parse_action(raw: str) -> dict:
    """Parse agent action JSON. Returns {} on failure."""
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    # Strip markdown fences
    stripped = raw.strip("`").strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except (json.JSONDecodeError, ValueError):
        pass
    return {}


def _action_matches(action_dict: dict, criteria: StepCriteria) -> bool:
    if action_dict.get("action") != criteria.action:
        return False
    if criteria.params and not _params_match(action_dict, criteria.params):
        return False
    return True


# ---------------------------------------------------------------------------
# State-check evaluation
# ---------------------------------------------------------------------------


def _evaluate_state_check(check: StateCheck, tracker: EpisodeTracker) -> bool:
    tracker_data: dict = {
        "flags_raised": tracker.flags_raised,
        "tool_calls": tracker.tool_calls,
        "actions_taken": tracker.actions_taken,
        "credited_steps": tracker._credited_steps,
        "step_count": tracker.step_count,
    }
    value = tracker_data.get(check.field)
    if value is None:
        return False
    if check.mode == "eq":
        return str(value) == str(check.expected)
    if check.mode == "contains":
        if isinstance(value, list):
            return check.expected in value
        return str(check.expected) in str(value)
    if check.mode == "gte":
        try:
            return float(value) >= float(check.expected)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
    return False


# ---------------------------------------------------------------------------
# TaskGrader
# ---------------------------------------------------------------------------


class TaskGrader:
    """Evaluates task completion and computes shaped rewards.

    Called from the environment on EVERY step() call (not only at terminal).
    """

    def grade(
        self,
        task: Task,
        tracker: EpisodeTracker,
        latest_step: StepRecord,
        hints_used: int = 0,
    ) -> GradeResult:
        """Grade one agent step. Dispatches based on populated SuccessCriteria fields."""
        criteria = task.success_criteria
        action_dict = latest_step.action_dict

        # --- Phishing safety check (pre-dispatch) ---
        if criteria.phishing_task and action_dict.get("action") != "flag_phishing":
            tracker.phishing_missed = True
            result = GradeResult(
                task_achieved=False,
                partial_progress=0.0,
                reason="SAFETY VIOLATION: phishing email not flagged",
                phishing_violation=True,
            )
            result.reward = 0.0
            return result

        # --- Dispatch by populated criteria fields (mirrors reference) ---
        if criteria.state_checks:
            result = self._grade_state_checks(criteria, tracker, latest_step)
        elif len(criteria.steps) > 1:
            result = self._grade_multi_step(criteria, tracker, latest_step)
        elif criteria.action_match is not None:
            result = self._grade_action_match(criteria, latest_step)
        elif len(criteria.steps) == 1:
            # Single-step hard task (e.g. flag_phishing with state_check already handled)
            result = self._grade_single_step(criteria.steps[0], tracker, latest_step)
        else:
            result = GradeResult(reason="no recognised success_criteria fields")

        # --- Compute shaped reward ---
        result.reward = self._compute_reward(result, tracker, hints_used)

        # --- Update monotonic previous_progress ---
        if result.partial_progress > tracker.previous_progress:
            tracker.previous_progress = result.partial_progress

        return result

    # ------------------------------------------------------------------
    # Grading strategies
    # ------------------------------------------------------------------

    def _grade_action_match(
        self,
        criteria: SuccessCriteria,
        latest_step: StepRecord,
    ) -> GradeResult:
        """Easy tier: check if the single action matches action_match criteria."""
        assert criteria.action_match is not None
        matched = _action_matches(latest_step.action_dict, criteria.action_match)
        achieved = matched and latest_step.valid
        return GradeResult(
            task_achieved=achieved,
            partial_progress=1.0 if achieved else 0.0,
            reason=f"action_match: matched={matched}, valid={latest_step.valid}",
        )

    def _grade_single_step(
        self,
        step: StepCriteria,
        tracker: EpisodeTracker,
        latest_step: StepRecord,
    ) -> GradeResult:
        """Single-step from steps list (used for hard phishing tasks)."""
        matched = _action_matches(latest_step.action_dict, step)
        achieved = matched and latest_step.valid
        return GradeResult(
            task_achieved=achieved,
            partial_progress=1.0 if achieved else 0.0,
            reason=f"single_step: action={step.action}, matched={matched}",
        )

    def _grade_multi_step(
        self,
        criteria: SuccessCriteria,
        tracker: EpisodeTracker,
        latest_step: StepRecord,
    ) -> GradeResult:
        """Medium tier: ordered multi-step sequence."""
        steps = criteria.steps
        total = len(steps)
        if total == 0:
            return GradeResult(reason="empty steps list")

        credited = tracker.get_credited_count()

        if credited >= total:
            return GradeResult(
                task_achieved=True,
                partial_progress=1.0,
                reason=f"multi_step: all {total} steps completed",
            )

        # Check next expected step
        next_step = steps[credited]
        if _action_matches(latest_step.action_dict, next_step) and latest_step.valid:
            step_key = f"step_{credited}_{next_step.action}"
            newly = tracker.credit_step(step_key)
            new_credited = credited + 1 if newly else credited
            progress = new_credited / total
            achieved = new_credited >= total
            return GradeResult(
                task_achieved=achieved,
                partial_progress=min(progress, 1.0),
                reason=f"multi_step: {new_credited}/{total} steps completed",
            )

        # Wrong step — return current partial progress
        return GradeResult(
            partial_progress=credited / total,
            reason=f"multi_step: expected step {credited + 1} '{next_step.action}', "
                   f"got '{latest_step.action_type}'",
        )

    def _grade_state_checks(
        self,
        criteria: SuccessCriteria,
        tracker: EpisodeTracker,
        latest_step: StepRecord,
    ) -> GradeResult:
        """Hard tier: multi-step sequence + end-state assertions.

        Steps provide the dense progress signal.
        State checks are the source of truth for task_achieved.
        """
        steps = criteria.steps
        total_steps = len(steps)
        credited = tracker.get_credited_count()

        # Advance through steps sequence
        if credited < total_steps:
            next_step = steps[credited]
            if _action_matches(latest_step.action_dict, next_step) and latest_step.valid:
                step_key = f"step_{credited}_{next_step.action}"
                newly = tracker.credit_step(step_key)
                if newly:
                    credited += 1

        step_progress = credited / total_steps if total_steps > 0 else 0.0

        # Evaluate state checks
        total_checks = len(criteria.state_checks)
        checks_passed = sum(
            1 for c in criteria.state_checks if _evaluate_state_check(c, tracker)
        )
        check_progress = checks_passed / total_checks if total_checks > 0 else 0.0

        # Combined progress: steps 70%, checks 30%
        if total_checks > 0:
            progress = step_progress * 0.7 + check_progress * 0.3
        else:
            progress = step_progress

        # Task achieved only when ALL state checks pass
        achieved = (checks_passed == total_checks and total_checks > 0) and (
            credited >= total_steps or total_steps == 0
        )

        return GradeResult(
            task_achieved=achieved,
            partial_progress=min(progress, 1.0),
            reason=(
                f"state_checks: {checks_passed}/{total_checks} passed, "
                f"steps: {credited}/{total_steps}"
            ),
        )

    # ------------------------------------------------------------------
    # Reward shaping
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        result: GradeResult,
        tracker: EpisodeTracker,
        hints_used: int = 0,
    ) -> float:
        """Compute shaped reward in [0.0, 0.99].

        inference.py clamps to [0.001, 0.999] so we never return exactly 0 or 1.
        """
        if result.phishing_violation or tracker.phishing_missed:
            return 0.0

        if result.task_achieved:
            base = 1.0
            return base * (HINT_DECAY ** hints_used)

        # Base: partial progress scaled to [0.0, 0.8]
        progress_reward = result.partial_progress * 0.8

        # Bonus for advancing progress (dense signal)
        progress_delta = result.partial_progress - tracker.previous_progress
        if progress_delta > 0:
            progress_reward += 0.1

        # Penalty for invalid actions
        if not (tracker.command_history[-1].valid if tracker.command_history else True):
            progress_reward *= 0.5

        # Hint decay
        if hints_used > 0:
            progress_reward *= HINT_DECAY ** hints_used

        # Clamp to [0.0, 0.99] — inference.py adds the 0.001 floor
        return min(max(progress_reward, 0.0), 0.99)

    # ------------------------------------------------------------------
    # Hints
    # ------------------------------------------------------------------

    def get_hint(self, task: Task, hints_used: int) -> str:
        """Return a progressive hint text (level determined by hints_used 0-indexed)."""
        if hints_used == 0:
            return task.hint_level_1 or "No hint available."
        if hints_used == 1:
            return task.hint_level_2 or "No hint available."
        return task.hint_level_3 or "No further hints available."
