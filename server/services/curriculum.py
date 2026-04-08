"""Curriculum learning manager for the Email Triage RL Environment."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field

from models import Difficulty, Task
from server.services.tasks import ALL_TASKS

logger = logging.getLogger(__name__)

MASTERY_WINDOW = 10
MASTERY_THRESHOLD = 0.70
FAST_TRACK_STREAK = 3
FAST_TRACK_RATE = 0.9

NOVELTY_BONUS = 100.0
WEAKNESS_WEIGHT = 50.0
SPACED_REP_BONUS = 30.0
RECENCY_PENALTY = 20.0

TIER_ORDER = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD]
TIER_MIN_EPISODES = {Difficulty.EASY: 6, Difficulty.MEDIUM: 6, Difficulty.HARD: 0}
TIER_ADVANCE_RATE = {Difficulty.EASY: 0.65, Difficulty.MEDIUM: 0.60, Difficulty.HARD: 1.0}


@dataclass
class TaskRecord:
    task_id: int
    attempts: int = 0
    successes: int = 0
    recent_results: list = field(default_factory=list)
    graduated: bool = False
    spaced_rep_interval: int = 3
    last_graduated_episode: int = 0
    last_attempted_episode: int = -10

    def success_rate(self) -> float:
        if not self.recent_results:
            return 0.0
        window = self.recent_results[-MASTERY_WINDOW:]
        weights = [0.85 ** (len(window) - i - 1) for i in range(len(window))]
        weighted = sum(w * int(r) for w, r in zip(weights, window))
        return weighted / sum(weights)


class Curriculum:
    """Manages task selection and tier progression."""

    def __init__(self) -> None:
        self.records: dict[int, TaskRecord] = {
            int(t.task_id): TaskRecord(task_id=int(t.task_id)) for t in ALL_TASKS
        }
        self._by_diff: dict[Difficulty, list[Task]] = defaultdict(list)
        for t in ALL_TASKS:
            self._by_diff[t.difficulty].append(t)

        self.current_difficulty: Difficulty = Difficulty.EASY
        self.episode_count: int = 0
        self.tier_episodes: int = 0
        self._fast_track_streak: int = 0

    def next_task(self) -> Task:
        pool = self._current_pool()
        if not pool:
            return random.choice(ALL_TASKS)
        scored = [(self._score(t), t) for t in pool]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]

    def record_result(self, task: Task, achieved: bool, reward: float) -> None:
        tid = int(task.task_id)
        rec = self.records[tid]
        rec.attempts += 1
        rec.last_attempted_episode = self.episode_count
        rec.recent_results.append(achieved)
        if achieved:
            rec.successes += 1

        self.episode_count += 1
        self.tier_episodes += 1

        if rec.attempts >= 3 and rec.success_rate() >= MASTERY_THRESHOLD and not rec.graduated:
            rec.graduated = True
            rec.last_graduated_episode = self.episode_count
            rec.spaced_rep_interval = 3

        if reward >= FAST_TRACK_RATE:
            self._fast_track_streak += 1
        else:
            self._fast_track_streak = 0

        self._maybe_promote()

    @property
    def chaos_probability(self) -> float:
        return 0.0

    def _current_pool(self) -> list[Task]:
        allowed = TIER_ORDER[: TIER_ORDER.index(self.current_difficulty) + 1]
        pool: list[Task] = []
        for d in allowed:
            pool.extend(self._by_diff[d])
        return pool

    def _score(self, task: Task) -> float:
        rec = self.records[int(task.task_id)]
        score = 0.0
        if rec.attempts == 0:
            score += NOVELTY_BONUS
        else:
            score += WEAKNESS_WEIGHT * (1.0 - rec.success_rate())
        if rec.graduated:
            elapsed = self.episode_count - rec.last_graduated_episode
            if elapsed >= rec.spaced_rep_interval:
                score += SPACED_REP_BONUS
        if self.episode_count - rec.last_attempted_episode <= 2:
            score -= RECENCY_PENALTY
        return score

    def _maybe_promote(self) -> None:
        current_idx = TIER_ORDER.index(self.current_difficulty)
        if current_idx >= len(TIER_ORDER) - 1:
            return

        if self._fast_track_streak >= FAST_TRACK_STREAK:
            self._promote()
            return

        min_ep = TIER_MIN_EPISODES[self.current_difficulty]
        if self.tier_episodes < min_ep:
            return

        pool = self._by_diff[self.current_difficulty]
        rates = [self.records[int(t.task_id)].success_rate() for t in pool]
        if rates and (sum(rates) / len(rates)) >= TIER_ADVANCE_RATE[self.current_difficulty]:
            self._promote()

    def _promote(self) -> None:
        current_idx = TIER_ORDER.index(self.current_difficulty)
        if current_idx < len(TIER_ORDER) - 1:
            self.current_difficulty = TIER_ORDER[current_idx + 1]
            self.tier_episodes = 0
            self._fast_track_streak = 0
