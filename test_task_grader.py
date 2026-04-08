"""Tests for TaskGrader — covers all three grading strategies and reward boundaries."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import Difficulty, EmailThread, StateCheck, StepCriteria, SuccessCriteria, Task, TaskID
from server.services.episode_tracker import EpisodeTracker
from server.services.task_grader import GradeResult, TaskGrader, _parse_action


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def grader():
    return TaskGrader()


@pytest.fixture
def tracker():
    return EpisodeTracker()


def _make_easy_task() -> Task:
    return Task(
        task_id=TaskID(0),
        difficulty=Difficulty.EASY,
        description="Classify billing email",
        email=EmailThread(sender="a@b.com", subject="Invoice", body="Wrong charge"),
        success_criteria=SuccessCriteria(
            action_match=StepCriteria(
                action="classify", params={"priority": "high", "category": "billing"}
            )
        ),
    )


def _make_medium_task() -> Task:
    return Task(
        task_id=TaskID(10),
        difficulty=Difficulty.MEDIUM,
        description="CRM lookup then route",
        email=EmailThread(sender="a@b.com", subject="Follow-up", body="Still waiting"),
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="use_tool", params={"tool": "crm"}),
                StepCriteria(action="assign_queue", params={"queue": "support"}),
            ]
        ),
    )


def _make_hard_task() -> Task:
    return Task(
        task_id=TaskID(21),
        difficulty=Difficulty.HARD,
        description="SLA breach pipeline",
        email=EmailThread(sender="a@b.com", subject="SLA", body="Breach risk"),
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="classify", params={"priority": "urgent"}),
                StepCriteria(action="use_tool", params={"tool": "ticketing"}),
                StepCriteria(action="escalate", params={"to": "manager"}),
            ],
            state_checks=[
                StateCheck(field="flags_raised", expected="sla_breach", mode="contains"),
                StateCheck(field="tool_calls", expected="ticketing", mode="contains"),
            ],
        ),
    )


def _make_phishing_task() -> Task:
    return Task(
        task_id=TaskID(2),
        difficulty=Difficulty.EASY,
        description="Flag phishing",
        email=EmailThread(sender="fake@paypa1.com", subject="Urgent", body="Click here"),
        success_criteria=SuccessCriteria(
            action_match=StepCriteria(action="flag_phishing"),
            phishing_task=True,
        ),
    )


def _record(tracker, raw_action):
    action_dict = _parse_action(raw_action)
    return tracker.record_step(raw_action, action_dict, bool(action_dict))


# ---------------------------------------------------------------------------
# Reward boundary tests
# ---------------------------------------------------------------------------

class TestRewardBoundaries:
    def test_reward_never_exactly_one_on_partial(self, grader, tracker):
        task = _make_easy_task()
        step = _record(tracker, '{"action": "classify", "priority": "low"}')  # wrong
        result = grader.grade(task, tracker, step)
        assert result.reward < 1.0

    def test_reward_at_most_one_on_completion(self, grader, tracker):
        task = _make_easy_task()
        step = _record(tracker, '{"action": "classify", "priority": "high", "category": "billing"}')
        result = grader.grade(task, tracker, step)
        assert result.task_achieved is True
        assert result.reward <= 1.0

    def test_reward_never_negative(self, grader, tracker):
        task = _make_easy_task()
        step = _record(tracker, '{"action": "wrong_action"}')
        result = grader.grade(task, tracker, step)
        assert result.reward >= 0.0

    def test_phishing_violation_gives_zero_reward(self, grader, tracker):
        task = _make_phishing_task()
        step = _record(tracker, '{"action": "classify", "priority": "high"}')
        result = grader.grade(task, tracker, step)
        assert result.reward == 0.0
        assert result.phishing_violation is True

    def test_invalid_action_json_gives_no_crash(self, grader, tracker):
        task = _make_easy_task()
        step = _record(tracker, "not json at all")
        result = grader.grade(task, tracker, step)
        assert isinstance(result.reward, float)
        assert result.reward >= 0.0


# ---------------------------------------------------------------------------
# Easy tier — action_match grading
# ---------------------------------------------------------------------------

class TestActionMatchGrading:
    def test_correct_action_achieves_task(self, grader, tracker):
        task = _make_easy_task()
        step = _record(tracker, '{"action": "classify", "priority": "high", "category": "billing"}')
        result = grader.grade(task, tracker, step)
        assert result.task_achieved is True
        assert result.partial_progress == 1.0

    def test_wrong_priority_does_not_achieve(self, grader, tracker):
        task = _make_easy_task()
        step = _record(tracker, '{"action": "classify", "priority": "low", "category": "billing"}')
        result = grader.grade(task, tracker, step)
        assert result.task_achieved is False
        assert result.partial_progress == 0.0

    def test_wrong_action_type_does_not_achieve(self, grader, tracker):
        task = _make_easy_task()
        step = _record(tracker, '{"action": "assign_queue", "queue": "billing"}')
        result = grader.grade(task, tracker, step)
        assert result.task_achieved is False

    def test_flag_phishing_achieves_phishing_task(self, grader, tracker):
        task = _make_phishing_task()
        step = _record(tracker, '{"action": "flag_phishing"}')
        result = grader.grade(task, tracker, step)
        assert result.task_achieved is True


# ---------------------------------------------------------------------------
# Medium tier — multi_step grading
# ---------------------------------------------------------------------------

class TestMultiStepGrading:
    def test_first_step_gives_partial_progress(self, grader, tracker):
        task = _make_medium_task()
        step = _record(tracker, '{"action": "use_tool", "tool": "crm"}')
        result = grader.grade(task, tracker, step)
        assert result.task_achieved is False
        assert result.partial_progress == pytest.approx(0.5)

    def test_wrong_first_step_gives_zero(self, grader, tracker):
        task = _make_medium_task()
        step = _record(tracker, '{"action": "assign_queue", "queue": "support"}')
        result = grader.grade(task, tracker, step)
        assert result.partial_progress == 0.0

    def test_both_steps_achieves_task(self, grader, tracker):
        task = _make_medium_task()
        # Step 1
        s1 = _record(tracker, '{"action": "use_tool", "tool": "crm"}')
        grader.grade(task, tracker, s1)
        # Step 2
        s2 = _record(tracker, '{"action": "assign_queue", "queue": "support"}')
        result = grader.grade(task, tracker, s2)
        assert result.task_achieved is True
        assert result.partial_progress == 1.0

    def test_duplicate_step_not_double_credited(self, grader, tracker):
        task = _make_medium_task()
        s1 = _record(tracker, '{"action": "use_tool", "tool": "crm"}')
        grader.grade(task, tracker, s1)
        credited_after_first = tracker.get_credited_count()
        # Same step again
        s2 = _record(tracker, '{"action": "use_tool", "tool": "crm"}')
        grader.grade(task, tracker, s2)
        assert tracker.get_credited_count() == credited_after_first


# ---------------------------------------------------------------------------
# Hard tier — state_checks grading
# ---------------------------------------------------------------------------

class TestStateChecksGrading:
    def test_no_state_checks_passed_gives_low_progress(self, grader, tracker):
        task = _make_hard_task()
        step = _record(tracker, '{"action": "classify", "priority": "urgent"}')
        result = grader.grade(task, tracker, step)
        assert result.task_achieved is False
        assert result.partial_progress > 0.0  # some progress for step 1

    def test_all_steps_and_checks_achieves_task(self, grader, tracker):
        task = _make_hard_task()
        # Step 1
        grader.grade(task, tracker, _record(tracker, '{"action": "classify", "priority": "urgent"}'))
        # Step 2 — ticketing also sets tool_calls
        grader.grade(task, tracker, _record(tracker, '{"action": "use_tool", "tool": "ticketing"}'))
        # Step 3 — escalate with sla_breach sets flag
        result = grader.grade(
            task,
            tracker,
            _record(tracker, '{"action": "escalate", "to": "manager", "reason": "sla_breach"}'),
        )
        assert result.task_achieved is True

    def test_state_check_fails_when_flag_missing(self, grader, tracker):
        task = _make_hard_task()
        grader.grade(task, tracker, _record(tracker, '{"action": "classify", "priority": "urgent"}'))
        grader.grade(task, tracker, _record(tracker, '{"action": "use_tool", "tool": "ticketing"}'))
        # Escalate WITHOUT sla_breach reason — flag not raised
        result = grader.grade(
            task,
            tracker,
            _record(tracker, '{"action": "escalate", "to": "manager", "reason": "just_urgent"}'),
        )
        assert result.task_achieved is False


# ---------------------------------------------------------------------------
# Monotonic progress
# ---------------------------------------------------------------------------

class TestMonotonicProgress:
    def test_progress_never_decreases(self, grader, tracker):
        task = _make_medium_task()
        s1 = _record(tracker, '{"action": "use_tool", "tool": "crm"}')
        r1 = grader.grade(task, tracker, s1)
        progress_after_step1 = tracker.previous_progress

        # Wrong step — progress should not go below previous
        s2 = _record(tracker, '{"action": "assign_queue", "queue": "wrong"}')
        r2 = grader.grade(task, tracker, s2)
        assert tracker.previous_progress >= progress_after_step1


# ---------------------------------------------------------------------------
# Grader called on every step (not only terminal)
# ---------------------------------------------------------------------------

class TestGraderCalledEveryStep:
    def test_grader_returns_result_on_every_step(self, grader, tracker):
        task = _make_medium_task()
        for action in [
            '{"action": "use_tool", "tool": "crm"}',
            '{"action": "assign_queue", "queue": "support"}',
        ]:
            step = _record(tracker, action)
            result = grader.grade(task, tracker, step)
            assert isinstance(result, GradeResult)
            assert isinstance(result.reward, float)


# ---------------------------------------------------------------------------
# Task count sanity check
# ---------------------------------------------------------------------------

class TestTaskDefinitions:
    def test_all_tasks_loaded(self):
        from server.services.tasks import ALL_TASKS, EASY_TASKS, MEDIUM_TASKS, HARD_TASKS
        assert len(EASY_TASKS) == 10
        assert len(MEDIUM_TASKS) == 10
        assert len(HARD_TASKS) == 10
        assert len(ALL_TASKS) == 30

    def test_all_task_ids_unique(self):
        from server.services.tasks import ALL_TASKS
        ids = [int(t.task_id) for t in ALL_TASKS]
        assert len(ids) == len(set(ids))

    def test_all_tasks_have_success_criteria(self):
        from server.services.tasks import ALL_TASKS
        for t in ALL_TASKS:
            sc = t.success_criteria
            has_criteria = (
                sc.action_match is not None
                or len(sc.steps) > 0
                or len(sc.state_checks) > 0
            )
            assert has_criteria, f"Task {t.task_id} has no success criteria"

    def test_hard_tasks_have_state_checks(self):
        from server.services.tasks import HARD_TASKS
        for t in HARD_TASKS:
            assert len(t.success_criteria.state_checks) > 0, \
                f"Hard task {t.task_id} missing state_checks"

    def test_medium_tasks_have_multi_step(self):
        from server.services.tasks import MEDIUM_TASKS
        for t in MEDIUM_TASKS:
            assert len(t.success_criteria.steps) >= 2, \
                f"Medium task {t.task_id} has fewer than 2 steps"

    def test_easy_tasks_have_action_match_or_single_step(self):
        from server.services.tasks import EASY_TASKS
        for t in EASY_TASKS:
            sc = t.success_criteria
            assert sc.action_match is not None, \
                f"Easy task {t.task_id} missing action_match"
