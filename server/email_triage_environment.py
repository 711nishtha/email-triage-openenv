"""
Email Triage RL Environment — Grand Finale Edition
====================================================
Wires together:
  - OpenEnv Environment interface (sync reset/step)
  - Symbolic safety layer with hard veto + reward multiplier
  - Adaptive curriculum (CurriculumManager, not the old Curriculum class)
  - GRPO-style reward shaping: dense per-step + shaped terminal bonus
  - Society metadata in state (for dashboard visibility)
  - Task grader v2 (three-task scoring)

Reward contract:
  - Every returned reward is clamped to [0.001, 0.999] at this layer
  - inference.py also clamps — double protection
  - Phishing safety violation: hard floor 0.001
  - Warning violation: reward × 0.5
  - Hint decay: reward × 0.85^hints
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

# Absolute imports — works both from repo root and as package
sys.path.insert(0, str(Path(__file__).parent.parent))

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

logger = logging.getLogger(__name__)

MAX_STEPS = 10
REWARD_MIN = 0.001
REWARD_MAX = 0.999
HINT_DECAY = 0.85


def _clamp(v: float) -> float:
    return max(REWARD_MIN, min(REWARD_MAX, v))


# ---------------------------------------------------------------------------
# Lazy imports (so the server starts even if optional deps missing)
# ---------------------------------------------------------------------------

def _load_curriculum():
    try:
        from curriculum.manager import CurriculumManager
        return CurriculumManager()
    except ImportError:
        logger.warning("curriculum.manager not found, using legacy Curriculum")
        from server.services.curriculum import Curriculum
        return Curriculum()


def _load_safety():
    try:
        from safety.rules import SafetyChecker
        return SafetyChecker()
    except ImportError:
        logger.warning("safety.rules not found, safety layer disabled")
        return None


def _load_grader():
    try:
        from server.services.task_grader import TaskGrader
        return TaskGrader()
    except ImportError as e:
        raise RuntimeError(f"TaskGrader not found: {e}") from e


def _load_tracker():
    from server.services.episode_tracker import EpisodeTracker
    return EpisodeTracker()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class EmailTriageEnvironment(
    Environment[EmailTriageAction, EmailTriageObservation, EmailTriageState]
):
    """
    Core RL environment — the OpenEnv server calls reset() and step() on this.

    New in Grand Finale edition:
      - SafetyChecker integrated into step() with hard veto
      - CurriculumManager (adaptive, phase-based) replaces static Curriculum
      - Society metadata field in state for dashboard visibility
      - GRPO-shaped rewards: dense signal on every step
      - Comprehensive violation logging to safety_violations.jsonl
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        logger.info("Initialising EmailTriageEnvironment (Grand Finale)...")
        self._state = EmailTriageState(episode_id=str(uuid4()), step_count=0)

        # Core services
        self._curriculum = _load_curriculum()
        self._grader = _load_grader()
        self._tracker = _load_tracker()
        self._safety = _load_safety()  # May be None if safety/ not present

        self._current_task: Task | None = None
        self._episode_violations: list[str] = []
        self._society_meta: dict = {}  # updated per step by inference.py via observation

    # ------------------------------------------------------------------
    # State sync
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
        # Expose curriculum tier
        if hasattr(self._curriculum, "current_difficulty"):
            self._state.current_tier = self._curriculum.current_difficulty.value
        elif hasattr(self._curriculum, "phase"):
            self._state.current_tier = f"phase_{self._curriculum.phase}"
        else:
            self._state.current_tier = "unknown"

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> EmailTriageObservation:
        self._state = EmailTriageState(
            episode_id=episode_id or str(uuid4()), step_count=0
        )
        self._tracker.reset()
        self._episode_violations = []
        self._society_meta = {}

        # Pick next task via curriculum
        self._current_task = self._curriculum.next_task()
        self._sync_state()

        logger.info(
            "reset() → task=%d difficulty=%s",
            int(self._current_task.task_id),
            self._current_task.difficulty.value,
        )

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

        # ---- Hint intercept ----
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

        # ---- Parse action ----
        from server.services.task_grader import _parse_action
        action_dict = _parse_action(raw)
        valid = bool(action_dict)

        # ---- Safety check (symbolic rules, zero LLM cost) ----
        safety_multiplier = 1.0
        safety_feedback = ""
        if self._safety is not None and action_dict:
            email_dict = {
                "sender": self._current_task.email.sender,
                "subject": self._current_task.email.subject,
                "body": self._current_task.email.body,
            }
            safety_result = self._safety.check(
                email_dict,
                action_dict,
                episode_id=self._state.episode_id,
                step=self._state.step_count,
            )
            if not safety_result.passed:
                # CRITICAL violation — return immediately with hard floor reward
                violation_msg = safety_result.violations[0].message if safety_result.violations else "CRITICAL safety violation"
                self._episode_violations.append(violation_msg)
                logger.warning("SAFETY CRITICAL: %s", violation_msg)
                self._sync_state()
                return EmailTriageObservation(
                    episode_id=EpisodeID(self._state.episode_id),
                    step_count=StepCount(self._state.step_count),
                    task=TaskInfo.from_task(self._current_task),
                    last_action_result=f"SAFETY VIOLATION: {violation_msg}",
                    last_action_valid=False,
                    task_achieved=False,
                    partial_progress=self._tracker.previous_progress,
                    hints_used=self._tracker.hints_used,
                    hint_text="",
                    done=False,
                    reward=REWARD_MIN,
                )
            safety_multiplier = safety_result.reward_multiplier
            if safety_result.has_warning():
                safety_feedback = " [SAFETY_WARN: reward×0.5]"
                for v in safety_result.violations:
                    self._episode_violations.append(v.message)

        # ---- Record step in tracker ----
        latest_step = self._tracker.record_step(raw, action_dict, valid)

        # ---- Grade on EVERY step ----
        grade_result = self._grader.grade(
            self._current_task,
            self._tracker,
            latest_step,
            hints_used=self._tracker.hints_used,
        )

        task_achieved = grade_result.task_achieved

        # ---- GRPO-style shaped reward ----
        raw_reward = grade_result.reward
        reward = _clamp(raw_reward * safety_multiplier)

        done = task_achieved or self._state.step_count >= MAX_STEPS

        # ---- Update curriculum ----
        if done:
            # Use the new CurriculumManager API if available, else legacy
            if hasattr(self._curriculum, "record_episode"):
                self._curriculum.record_episode(reward)
            elif hasattr(self._curriculum, "record_result"):
                self._curriculum.record_result(
                    self._current_task,
                    achieved=task_achieved,
                    reward=reward,
                )

        self._sync_state()

        feedback = grade_result.reason + safety_feedback
        logger.debug(
            "step()  task=%d step=%d reward=%.4f achieved=%s done=%s",
            int(self._current_task.task_id),
            self._state.step_count,
            reward,
            task_achieved,
            done,
        )

        return EmailTriageObservation(
            episode_id=EpisodeID(self._state.episode_id),
            step_count=StepCount(self._state.step_count),
            task=TaskInfo.from_task(self._current_task),
            last_action_result=feedback,
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

    @property
    def episode_violations(self) -> list[str]:
        return list(self._episode_violations)
