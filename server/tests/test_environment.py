"""Tests for the EmailTriageEnvironment reset/step/state interface."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from models import EmailTriageAction, EmailTriageObservation
from server.email_triage_environment import EmailTriageEnvironment


@pytest.fixture
def env():
    return EmailTriageEnvironment()


class TestReset:
    def test_reset_returns_observation(self, env):
        obs = env.reset()
        assert isinstance(obs, EmailTriageObservation)

    def test_reset_has_task(self, env):
        obs = env.reset()
        assert obs.task is not None

    def test_reset_step_count_zero(self, env):
        obs = env.reset()
        assert obs.step_count == 0

    def test_reset_not_done(self, env):
        obs = env.reset()
        assert obs.done is False

    def test_reset_zero_reward(self, env):
        obs = env.reset()
        assert obs.reward == 0.0


class TestStep:
    def test_step_requires_reset_first(self, env):
        with pytest.raises(AssertionError):
            env.step(EmailTriageAction(action='{"action": "classify", "priority": "high"}'))

    def test_step_returns_observation(self, env):
        env.reset()
        obs = env.step(EmailTriageAction(action='{"action": "classify", "priority": "high", "category": "billing"}'))
        assert isinstance(obs, EmailTriageObservation)

    def test_step_increments_step_count(self, env):
        env.reset()
        obs = env.step(EmailTriageAction(action='{"action": "classify", "priority": "medium"}'))
        assert obs.step_count == 1

    def test_step_reward_is_float(self, env):
        env.reset()
        obs = env.step(EmailTriageAction(action='{"action": "classify", "priority": "high"}'))
        assert isinstance(obs.reward, float)

    def test_step_hint_request(self, env):
        env.reset()
        obs = env.step(EmailTriageAction(action="hint"))
        assert obs.hints_used == 1
        assert obs.hint_text != ""

    def test_step_done_after_max_steps(self, env):
        from server.email_triage_environment import MAX_STEPS
        env.reset()
        obs = None
        for _ in range(MAX_STEPS):
            obs = env.step(EmailTriageAction(action='{"action": "classify", "priority": "low"}'))
        assert obs.done is True

    def test_step_done_on_task_achieved(self, env):
        # Force task 0 (easy classify billing)
        from server.services.tasks import TASKS_BY_ID
        env.reset()
        env._current_task = TASKS_BY_ID[0]
        obs = env.step(
            EmailTriageAction(action='{"action": "classify", "priority": "high", "category": "billing"}')
        )
        assert obs.task_achieved is True
        assert obs.done is True


class TestState:
    def test_state_accessible_after_reset(self, env):
        env.reset()
        state = env.state
        assert state is not None
        assert state.current_task is not None

    def test_state_tier_is_string(self, env):
        env.reset()
        assert isinstance(env.state.current_tier, str)
