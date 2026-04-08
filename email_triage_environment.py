"""
Email Triage RL Environment Implementation.

Mirrors the reference AwsRlEnvironment structure:
  - Inherits from openenv.core.env_server.interfaces.Environment
  - reset() and step() are SYNCHRONOUS (not async)
  - Grader called on EVERY step() with (task, tracker, latest_step, hints_used)
  - GradeResult.reward stored and monotonic progress updated
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from models import (
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
    EpisodeID,
    StepCount,
    Task,
    TaskInfo,
    TrackerState,
)
from server.services.curriculum import Curriculum
from server.services.episode_tracker import EpisodeTracker
from server.services.task_grader import TaskGrader, _parse_action

logger = logging.getLogger(__name__)

MAX_STEPS = 10


class EmailTriageEnvironment(
    Environment[EmailTriageAction, EmailTriageObservation, EmailTriageState]
):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        logger.info("Initialising Email Triage Environment...")
        self._state = EmailTriageState(episode_id=str(uuid4()), step_count=0)
        self._curriculum = Curriculum()
        self._grader = TaskGrader()
        self._tracker = EpisodeTracker()
        self._current_task: Task | None = None

    # ------------------------------------------------------------------
    # State sync helper
    # ------------------------------------------------------------------

    def _sync_state(self) -> None:
        self._state.current_task = self._current_task
        self._state.tracker = TrackerState(
            step_count=self._tracker.step_count,
            hints_used=self._tracker.hints_used,
            progress=self._tracker.previous_progress,
            actions_taken=list(self._tracker.actions_taken),
            credited_steps=list(self._tracker._credited_steps),
            phishing_missed=self._tracker.phishing_missed,
            tool_calls=list(self._tracker.tool_calls),
            flags_raised=list(self._tracker.flags_raised),
        )
        self._state.current_tier = self._curriculum.current_difficulty.value

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EmailTriageObservation:
        self._state = EmailTriageState(episode_id=episode_id or str(uuid4()), step_count=0)
        self._tracker.reset()
        self._current_task = self._curriculum.next_task()
        self._sync_state()

        return EmailTriageObservation(
            episode_id=EpisodeID(self._state.episode_id),
            step_count=StepCount(0),
            task=TaskInfo.from_task(self._current_task),
            last_action_result=(
                f"New episode. Task [{self._current_task.difficulty.value.upper()}]: "
                f"{self._current_task.description}"
            ),
            last_action_valid=True,
            task_achieved=False,
            partial_progress=0.0,
            hints_used=0,
            hint_text="",
            done=False,
            reward=0.0,
        )

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(
        self,
        action: EmailTriageAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> EmailTriageObservation:
        assert self._current_task is not None, "Call reset() before step()"
        self._state.step_count += 1

        raw = action.action.strip()

        # --- Intercept hint request ---
        if raw.lower() in ("hint", "get_hint", '{"action": "hint"}'):
            hint_text = self._grader.get_hint(self._current_task, self._tracker.hints_used)
            self._tracker.record_hint()
            self._sync_state()
            return EmailTriageObservation(
                episode_id=EpisodeID(self._state.episode_id),
                step_count=StepCount(self._state.step_count),
                task=TaskInfo.from_task(self._current_task),
                last_action_result=f"HINT [{self._tracker.hints_used}]: {hint_text}",
                last_action_valid=True,
                task_achieved=False,
                partial_progress=self._tracker.previous_progress,
                hints_used=self._tracker.hints_used,
                hint_text=hint_text,
                done=False,
                reward=0.0,
            )

        # --- Parse and record action ---
        action_dict = _parse_action(raw)
        valid = bool(action_dict)
        latest_step = self._tracker.record_step(raw, action_dict, valid)

        # --- Grade on EVERY step (not only terminal) ---
        grade_result = self._grader.grade(
            self._current_task,
            self._tracker,
            latest_step,
            hints_used=self._tracker.hints_used,
        )

        task_achieved = grade_result.task_achieved
        reward = grade_result.reward

        if task_achieved:
            self._curriculum.record_result(self._current_task, achieved=True, reward=reward)

        done = task_achieved or self._state.step_count >= MAX_STEPS
        if done and not task_achieved:
            self._curriculum.record_result(self._current_task, achieved=False, reward=reward)

        self._sync_state()

        return EmailTriageObservation(
            episode_id=EpisodeID(self._state.episode_id),
            step_count=StepCount(self._state.step_count),
            task=TaskInfo.from_task(self._current_task),
            last_action_result=grade_result.reason,
            last_action_valid=valid,
            task_achieved=task_achieved,
            partial_progress=self._tracker.previous_progress,
            hints_used=self._tracker.hints_used,
            hint_text="",
            done=done,
            reward=reward,
        )

    # ------------------------------------------------------------------
    # State property
    # ------------------------------------------------------------------

    @property
    def state(self) -> EmailTriageState:
        return self._state
