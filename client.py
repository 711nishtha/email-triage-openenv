"""Email Triage Environment Client."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from models import (
    Difficulty,
    EmailTriageAction,
    EmailTriageObservation,
    EmailTriageState,
    EmailThread,
    EpisodeID,
    StepCount,
    TaskID,
    TaskInfo,
)


class EmailTriageEnv(EnvClient[EmailTriageAction, EmailTriageObservation, EmailTriageState]):
    """
    Client for the Email Triage RL Environment.

    Maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.

    Example:
        >>> with EmailTriageEnv(base_url="http://localhost:7860") as client:
        ...     result = client.reset()
        ...     print(result.observation.task.email_subject)
        ...     result = client.step(EmailTriageAction(action='{"action": "flag_phishing"}'))
        ...     print(result.reward)

    Example with Docker:
        >>> client = EmailTriageEnv.from_docker_image("email-triage-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(EmailTriageAction(action='{"action": "classify", "priority": "high"}'))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: EmailTriageAction) -> Dict:
        return {"action": action.action}

    def _parse_result(self, payload: Dict) -> StepResult[EmailTriageObservation]:
        obs_data = payload.get("observation", {})
        task_data = obs_data.get("task")

        task: TaskInfo | None = None
        if task_data:
            thread_raw = task_data.get("thread_history", [])
            thread = []
            for msg in thread_raw:
                if isinstance(msg, dict):
                    thread.append(EmailThread(**msg))
            task = TaskInfo(
                task_id=TaskID(int(task_data.get("task_id", 0))),
                difficulty=Difficulty(task_data.get("difficulty", "easy")),
                description=task_data.get("description", ""),
                email_subject=task_data.get("email_subject", ""),
                email_sender=task_data.get("email_sender", ""),
                email_body=task_data.get("email_body", ""),
                thread_history=thread,
            )

        observation = EmailTriageObservation(
            episode_id=EpisodeID(obs_data.get("episode_id", "")),
            step_count=StepCount(obs_data.get("step_count", 0)),
            task=task,
            last_action_result=obs_data.get("last_action_result", ""),
            last_action_valid=obs_data.get("last_action_valid", True),
            task_achieved=obs_data.get("task_achieved", False),
            partial_progress=obs_data.get("partial_progress", 0.0),
            hints_used=obs_data.get("hints_used", 0),
            hint_text=obs_data.get("hint_text", ""),
            done=payload.get("done", False),
            reward=payload.get("reward", 0.0),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> EmailTriageState:
        return EmailTriageState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            current_tier=payload.get("current_tier", "easy"),
        )
