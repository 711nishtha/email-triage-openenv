"""
curriculum/manager.py — Adaptive Curriculum Manager (Grand Finale)
====================================================================
Controls task difficulty, email pool, and phishing ratio as training progresses.

Phase progression
-----------------
  Phase 0 (warmup)      : Easy tasks only, low phishing ratio (build confidence)
  Phase 1 (standard)    : Easy + Medium tasks, standard phishing ratio
  Phase 2 (mixed)       : All tasks, higher phishing ratio
  Phase 3 (adversarial) : All tasks, adversarial difficulty, high phishing ratio

Promotion triggers
------------------
  Fast-track : 3 consecutive episodes with score >= FAST_TRACK_THRESHOLD (0.88)
  Mastery    : rolling avg >= MASTERY_THRESHOLD (0.72) over MASTERY_WINDOW (8) episodes
  Regression : if avg drops below REGRESSION_THRESHOLD (0.35) for REGRESSION_WINDOW (5)
               episodes, phase decreases by 1 (prevents overfitting to easy tasks)

GRPO signal tracking
---------------------
  Records per-episode (score, safety_violations, phishing_caught) for
  offline GRPO/PPO training signal. Exposes get_grpo_batch() for trainers.

Dual API
---------
  New API  : used by Grand Finale EmailTriageEnvironment
    cm.current_schedule()          → dict with phase config
    cm.record_episode(score, ...)  → bool (True if phase changed)
    cm.get_stats()                 → dict for dashboard

  Legacy API : compatible with server.services.curriculum.Curriculum
    cm.next_task()                 → Task (adaptive selection)
    cm.record_result(task, ...)    → None

Both APIs share internal state.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (all overridable via environment variables)
# ---------------------------------------------------------------------------

MASTERY_THRESHOLD: float = float(os.getenv("MASTERY_THRESHOLD", "0.72"))
MASTERY_WINDOW: int = int(os.getenv("MASTERY_WINDOW", "8"))
FAST_TRACK_THRESHOLD: float = float(os.getenv("FAST_TRACK_THRESHOLD", "0.88"))
FAST_TRACK_STREAK: int = int(os.getenv("FAST_TRACK_STREAK", "3"))
REGRESSION_THRESHOLD: float = float(os.getenv("REGRESSION_THRESHOLD", "0.35"))
REGRESSION_WINDOW: int = int(os.getenv("REGRESSION_WINDOW", "5"))
ALLOW_REGRESSION: bool = os.getenv("ALLOW_REGRESSION", "true").lower() == "true"

CURRICULUM_LOG = Path(os.getenv("CURRICULUM_LOG", "curriculum_log.jsonl"))

PHASE_SCHEDULES: dict[int, dict] = {
    0: {
        "name": "warmup",
        "tasks": [1],
        "difficulty": "warmup",
        "phishing_ratio": 0.15,
        "description": "Easy classification tasks — build confidence",
    },
    1: {
        "name": "standard",
        "tasks": [1, 2],
        "difficulty": "standard",
        "phishing_ratio": 0.20,
        "description": "Label + action — introduce multi-step workflows",
    },
    2: {
        "name": "mixed",
        "tasks": [1, 2, 3],
        "difficulty": "mixed",
        "phishing_ratio": 0.25,
        "description": "Full three-task scoring with thread history",
    },
    3: {
        "name": "adversarial",
        "tasks": [1, 2, 3],
        "difficulty": "adversarial",
        "phishing_ratio": 0.35,
        "description": "Adversarial phishing, SLA breaches, subtle attacks",
    },
}

MAX_PHASE = max(PHASE_SCHEDULES.keys())


# ---------------------------------------------------------------------------
# Episode record
# ---------------------------------------------------------------------------


@dataclass
class EpisodeRecord:
    episode: int
    phase: int
    task_number: int
    score: float
    achieved: bool
    safety_violations: int = 0
    phishing_caught: int = 0
    phishing_missed: int = 0
    hints_used: int = 0


# ---------------------------------------------------------------------------
# CurriculumManager
# ---------------------------------------------------------------------------


class CurriculumManager:
    """
    Adaptive curriculum with mastery-based promotion and regression detection.

    Fully compatible with both the Grand Finale environment and the legacy
    server.services.curriculum.Curriculum interface.
    """

    def __init__(self) -> None:
        self._phase: int = 0
        self._episode_count: int = 0
        self._phase_episodes: int = 0
        self._fast_track_streak: int = 0
        self._regression_streak: int = 0
        self._recent_scores: deque[float] = deque(maxlen=max(MASTERY_WINDOW, REGRESSION_WINDOW))
        self._all_records: list[EpisodeRecord] = []
        self._promotions: list[str] = []
        self._regressions: list[str] = []

        # Legacy curriculum state
        self._legacy_task_records: dict[int, dict] = {}
        self._legacy_episode: int = 0

        logger.info("CurriculumManager initialised at phase 0 (warmup)")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def phase(self) -> int:
        return self._phase

    @property
    def episode(self) -> int:
        return self._episode_count

    # ------------------------------------------------------------------
    # New API — used by Grand Finale EmailTriageEnvironment
    # ------------------------------------------------------------------

    def current_schedule(self) -> dict:
        """Return the current phase configuration dict."""
        return PHASE_SCHEDULES[min(self._phase, MAX_PHASE)]

    def current_task_number(self) -> int:
        """Return which task number (1, 2, or 3) to run this episode."""
        tasks = self.current_schedule()["tasks"]
        return tasks[self._phase_episodes % len(tasks)]

    def record_episode(
        self,
        score: float,
        achieved: bool = False,
        safety_violations: int = 0,
        phishing_caught: int = 0,
        phishing_missed: int = 0,
        hints_used: int = 0,
        task_number: Optional[int] = None,
    ) -> bool:
        """
        Record one episode result and check for phase transition.

        Returns True if the agent was promoted or regressed.
        """
        task_num = task_number if task_number is not None else self.current_task_number()
        record = EpisodeRecord(
            episode=self._episode_count,
            phase=self._phase,
            task_number=task_num,
            score=score,
            achieved=achieved,
            safety_violations=safety_violations,
            phishing_caught=phishing_caught,
            phishing_missed=phishing_missed,
            hints_used=hints_used,
        )
        self._all_records.append(record)
        self._recent_scores.append(score)
        self._episode_count += 1
        self._phase_episodes += 1

        # Fast-track streak tracking
        if score >= FAST_TRACK_THRESHOLD:
            self._fast_track_streak += 1
        else:
            self._fast_track_streak = 0

        # Regression streak tracking
        if score < REGRESSION_THRESHOLD:
            self._regression_streak += 1
        else:
            self._regression_streak = 0

        self._log_episode(record)
        return self._check_transitions()

    def _check_transitions(self) -> bool:
        """Check for promotion or regression. Returns True if a transition occurred."""
        # Fast-track promotion
        if self._fast_track_streak >= FAST_TRACK_STREAK and self._phase < MAX_PHASE:
            self._promote(f"fast-track ({self._fast_track_streak}× score≥{FAST_TRACK_THRESHOLD})")
            return True

        # Mastery promotion
        if len(self._recent_scores) >= MASTERY_WINDOW and self._phase < MAX_PHASE:
            recent = list(self._recent_scores)[-MASTERY_WINDOW:]
            avg = sum(recent) / len(recent)
            if avg >= MASTERY_THRESHOLD:
                self._promote(f"mastery (avg={avg:.3f}≥{MASTERY_THRESHOLD} over {MASTERY_WINDOW} eps)")
                return True

        # Regression demotion
        if ALLOW_REGRESSION and self._regression_streak >= REGRESSION_WINDOW and self._phase > 0:
            recent = list(self._recent_scores)[-REGRESSION_WINDOW:]
            avg = sum(recent) / len(recent)
            self._demote(f"regression (avg={avg:.3f}<{REGRESSION_THRESHOLD} for {REGRESSION_WINDOW} eps)")
            return True

        return False

    def _promote(self, reason: str) -> None:
        old = self._phase
        self._phase = min(self._phase + 1, MAX_PHASE)
        self._fast_track_streak = 0
        self._regression_streak = 0
        self._phase_episodes = 0
        self._recent_scores.clear()
        msg = f"PROMOTION: Phase {old} → {self._phase} ({reason})"
        self._promotions.append(msg)
        logger.info(msg)

    def _demote(self, reason: str) -> None:
        old = self._phase
        self._phase = max(self._phase - 1, 0)
        self._regression_streak = 0
        self._fast_track_streak = 0
        self._phase_episodes = 0
        self._recent_scores.clear()
        msg = f"REGRESSION: Phase {old} → {self._phase} ({reason})"
        self._regressions.append(msg)
        logger.warning(msg)

    def _log_episode(self, record: EpisodeRecord) -> None:
        entry = {
            "episode": record.episode,
            "phase": record.phase,
            "task": record.task_number,
            "score": round(record.score, 4),
            "achieved": record.achieved,
            "safety_violations": record.safety_violations,
            "phishing_caught": record.phishing_caught,
            "phishing_missed": record.phishing_missed,
        }
        try:
            with open(CURRICULUM_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass

    def get_stats(self) -> dict[str, Any]:
        """Return dashboard-ready statistics dict."""
        recent = list(self._recent_scores)
        recent_avg = sum(recent) / len(recent) if recent else 0.0

        # Per-phase score averages
        by_phase: dict[int, list[float]] = {}
        for rec in self._all_records:
            by_phase.setdefault(rec.phase, []).append(rec.score)
        phase_avgs = {p: round(sum(s) / len(s), 4) for p, s in by_phase.items()}

        # Safety aggregates
        total_violations = sum(r.safety_violations for r in self._all_records)
        total_phishing_caught = sum(r.phishing_caught for r in self._all_records)
        total_phishing_missed = sum(r.phishing_missed for r in self._all_records)

        return {
            "phase": self._phase,
            "phase_name": PHASE_SCHEDULES[min(self._phase, MAX_PHASE)]["name"],
            "episode": self._episode_count,
            "phase_episodes": self._phase_episodes,
            "recent_avg": round(recent_avg, 4),
            "fast_track_streak": self._fast_track_streak,
            "regression_streak": self._regression_streak,
            "promotions": self._promotions[-5:],   # last 5 only for JSON size
            "regressions": self._regressions[-5:],
            "phase_score_avgs": phase_avgs,
            "total_episodes": len(self._all_records),
            "safety_violations": total_violations,
            "phishing_caught": total_phishing_caught,
            "phishing_missed": total_phishing_missed,
            "schedule": self.current_schedule(),
        }

    def get_grpo_batch(self, n: int = 100) -> list[dict]:
        """Return the last n episode records as dicts for offline GRPO training."""
        return [
            {
                "episode": r.episode,
                "phase": r.phase,
                "task": r.task_number,
                "score": r.score,
                "achieved": r.achieved,
            }
            for r in self._all_records[-n:]
        ]

    # ------------------------------------------------------------------
    # Legacy API — compatible with server.services.curriculum.Curriculum
    # ------------------------------------------------------------------

    def next_task(self):
        """
        Legacy API: return the next Task object (adaptive selection).

        Selects from the current phase's difficulty pool, prioritizing:
          1. Never-attempted tasks (novelty bonus)
          2. Tasks with low success rate (weakness targeting)
          3. Avoiding recently seen tasks (spaced repetition)
        """
        try:
            from server.services.tasks import ALL_TASKS
        except ImportError:
            raise RuntimeError("server.services.tasks not available — check your installation")

        schedule = self.current_schedule()
        difficulty_name = schedule["difficulty"]

        # Map phase difficulty to task difficulty filter
        _DIFF_MAP = {
            "warmup": ["easy"],
            "standard": ["easy"],
            "mixed": ["easy", "medium"],
            "adversarial": ["easy", "medium", "hard"],
        }
        allowed_difficulties = _DIFF_MAP.get(difficulty_name, ["easy"])

        # Build candidate pool
        if difficulty_name == "adversarial":
            pool = ALL_TASKS  # all difficulties in adversarial phase
        else:
            pool = [t for t in ALL_TASKS if t.difficulty.value in allowed_difficulties]

        if not pool:
            pool = ALL_TASKS

        # Score each task: novelty + weakness + recency avoidance
        scored = []
        for task in pool:
            tid = int(task.task_id)
            rec = self._legacy_task_records.get(
                tid, {"attempts": 0, "successes": 0, "last": -999}
            )

            novelty_bonus = 100.0 if rec["attempts"] == 0 else 0.0
            success_rate = rec["successes"] / max(rec["attempts"], 1)
            weakness_score = 50.0 * (1.0 - success_rate)
            recency_penalty = max(0.0, 20.0 - (self._legacy_episode - rec["last"]) * 2)

            total_score = novelty_bonus + weakness_score - recency_penalty
            scored.append((total_score, task))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def record_result(self, task, achieved: bool, reward: float) -> None:
        """Legacy API: record task result without full episode tracking."""
        tid = int(task.task_id)
        if tid not in self._legacy_task_records:
            self._legacy_task_records[tid] = {"attempts": 0, "successes": 0, "last": -999}
        rec = self._legacy_task_records[tid]
        rec["attempts"] += 1
        rec["last"] = self._legacy_episode
        if achieved:
            rec["successes"] += 1
        self._legacy_episode += 1

        # Delegate to new API for phase tracking
        self.record_episode(score=reward, achieved=achieved)

    # ------------------------------------------------------------------
    # Legacy property compatibility
    # ------------------------------------------------------------------

    @property
    def current_difficulty(self):
        """
        Legacy attribute returning a Difficulty-like enum value.
        Used by server/app.py society_stats endpoint.
        """
        from enum import Enum

        class _FakeDiff(str, Enum):
            EASY = "easy"
            MEDIUM = "medium"
            HARD = "hard"

        schedule = self.current_schedule()
        diff = schedule.get("difficulty", "warmup")
        if diff in ("warmup", "standard"):
            return _FakeDiff.EASY
        elif diff == "mixed":
            return _FakeDiff.MEDIUM
        else:
            return _FakeDiff.HARD
