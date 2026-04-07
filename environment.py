"""
environment.py — Core email triage environment logic.

Implements the OpenEnv API:
  - reset(task_id) -> Observation
  - step(action) -> (Observation, reward, done, info)
  - state() -> EnvironmentState
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import time

from models import (
    Action,
    ActionType,
    Observation,
    EnvironmentState,
    EmailWithGroundTruth,
    ToolResult,
)
from data import get_emails_for_task, TASK_MAX_STEPS
from tools import execute_tool, AVAILABLE_TOOLS
from rewards import compute_reward


class EmailTriageEnvironment:
    """
    Stateful email triage environment.
    One instance per session (or per reset).
    """

    def __init__(self) -> None:
        self.task_id: str = "easy"
        self.emails: List[EmailWithGroundTruth] = []
        self.inbox: List[Dict[str, Any]] = []       # What the agent sees
        self.triaged: List[Dict[str, Any]] = []     # Completed triage records
        self.triaged_actions: List[Dict[str, Any]] = []  # For grading
        self.step_count: int = 0
        self.max_steps: int = 10
        self.done: bool = False
        self.cumulative_reward: float = 0.0
        self.tool_results: List[Dict[str, Any]] = []
        self._email_map: Dict[str, EmailWithGroundTruth] = {}

    # ─────────────────────────────────────────
    # reset()
    # ─────────────────────────────────────────

    def reset(self, task_id: str = "easy") -> Observation:
        """Reset environment for a new task. Returns initial observation."""
        self.task_id = task_id
        self.emails = get_emails_for_task(task_id)
        self.max_steps = TASK_MAX_STEPS[task_id]

        # Build agent-visible inbox (no ground truth)
        self.inbox = [e.to_observation() for e in self.emails]
        self.triaged = []
        self.triaged_actions = []
        self.step_count = 0
        self.done = False
        self.cumulative_reward = 0.0
        self.tool_results = []
        self._email_map = {e.id: e for e in self.emails}

        return self._build_observation(message=f"Environment reset. Task: {task_id}. You have {len(self.inbox)} emails to triage.")

    # ─────────────────────────────────────────
    # step()
    # ─────────────────────────────────────────

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Process an agent action and return (observation, reward, done, info).
        """
        if self.done:
            obs = self._build_observation(message="Episode already complete. Call /reset to start a new episode.")
            return obs, 0.0, True, {"error": "Episode already done"}

        self.step_count += 1
        reward = 0.0
        info: Dict[str, Any] = {"action_type": action.action_type, "step": self.step_count}

        # ── done action ─────────────────────────────────────
        if action.action_type == ActionType.done:
            reward_info = compute_reward(
                action_type="done",
                email=None,
                action_category=None,
                action_priority=None,
                action_route=None,
                is_done=True,
                task_emails=self.emails,
                triaged_actions=self.triaged_actions,
                step_count=self.step_count,
                max_steps=self.max_steps,
            )
            reward = reward_info["reward"]
            # Clamp reward to open interval (0,1) – never exact 0.0 or 1.0
            if reward <= 0.0:
                reward = 0.001
            elif reward >= 1.0:
                reward = 0.999
            self.done = True
            self.cumulative_reward = reward
            info["episode_summary"] = reward_info.get("episode_summary")
            obs = self._build_observation(message=f"Episode complete. Final score: {reward:.4f}")
            return obs, reward, True, info

        # ── use_tool action ──────────────────────────────────
        if action.action_type == ActionType.use_tool:
            result = execute_tool(
                tool_name=action.tool_name or "",
                tool_args=action.tool_args or {},
            )
            tool_result = {
                "step": self.step_count,
                "tool": action.tool_name,
                "args": action.tool_args,
                "result": result,
            }
            self.tool_results.append(tool_result)
            info["tool_result"] = result
            done = self._check_done()
            obs = self._build_observation(
                message=f"Tool '{action.tool_name}' executed.",
                last_tool_result=result,
            )
            # Tool reward is always 0.001 (never 0)
            tool_reward = 0.001
            return obs, tool_reward, done, info

        # ── escalate action ──────────────────────────────────
        if action.action_type == ActionType.escalate:
            email_id = action.email_id or ""
            email = self._email_map.get(email_id)
            if email is None:
                info["error"] = f"Email ID '{email_id}' not found"
                obs = self._build_observation(message=f"Error: Email '{email_id}' not found.")
                return obs, 0.001, False, info

            triage_record = {
                "email_id": email_id,
                "action": "escalate",
                "priority": (action.priority.value if action.priority else "urgent"),
                "category": (action.category.value if action.category else "urgent_business"),
                "route_to": (action.route_to.value if action.route_to else "executive"),
                "step": self.step_count,
            }
            self._record_triage(email_id, triage_record)
            info["escalated"] = email_id

            reward_info = compute_reward(
                action_type="triage",
                email=email,
                action_category=triage_record["category"],
                action_priority=triage_record["priority"],
                action_route=triage_record["route_to"],
                is_done=False,
                task_emails=self.emails,
                triaged_actions=self.triaged_actions,
                step_count=self.step_count,
                max_steps=self.max_steps,
            )
            reward = reward_info["reward"]
            # Clamp
            if reward <= 0.0:
                reward = 0.001
            elif reward >= 1.0:
                reward = 0.999
            self.cumulative_reward += reward
            obs = self._build_observation(message=f"Email '{email_id}' escalated.")
            return obs, reward, self._check_done(), info

        # ── triage action ────────────────────────────────────
        if action.action_type == ActionType.triage:
            email_id = action.email_id or ""
            email = self._email_map.get(email_id)
            if email is None:
                info["error"] = f"Email ID '{email_id}' not found"
                obs = self._build_observation(message=f"Error: Email '{email_id}' not found.")
                return obs, 0.001, False, info

            # Already triaged?
            already_triaged = any(a["email_id"] == email_id for a in self.triaged_actions)
            if already_triaged:
                info["warning"] = f"Email '{email_id}' already triaged"
                obs = self._build_observation(message=f"Warning: Email '{email_id}' was already triaged.")
                return obs, 0.001, False, info

            triage_record = {
                "email_id": email_id,
                "action": "triage",
                "priority": (action.priority.value if action.priority else "medium"),
                "category": (action.category.value if action.category else "internal_task"),
                "route_to": (action.route_to.value if action.route_to else "manager"),
                "step": self.step_count,
            }
            self._record_triage(email_id, triage_record)

            reward_info = compute_reward(
                action_type="triage",
                email=email,
                action_category=triage_record["category"],
                action_priority=triage_record["priority"],
                action_route=triage_record["route_to"],
                is_done=False,
                task_emails=self.emails,
                triaged_actions=self.triaged_actions,
                step_count=self.step_count,
                max_steps=self.max_steps,
            )
            reward = reward_info["reward"]
            # Clamp
            if reward <= 0.0:
                reward = 0.001
            elif reward >= 1.0:
                reward = 0.999
            self.cumulative_reward += reward

            remaining = len(self.inbox)
            done = self._check_done()
            obs = self._build_observation(
                message=f"Email '{email_id}' triaged. {remaining} emails remaining in inbox."
            )
            return obs, reward, done, info

        # Unknown action type
        info["error"] = f"Unknown action_type: {action.action_type}"
        obs = self._build_observation(message=f"Error: Unknown action type.")
        return obs, 0.001, False, info

    # ─────────────────────────────────────────
    # state()
    # ─────────────────────────────────────────

    def state(self) -> EnvironmentState:
        """Return the full environment state."""
        return EnvironmentState(
            task_id=self.task_id,
            step_count=self.step_count,
            max_steps=self.max_steps,
            inbox=self.inbox,
            triaged=self.triaged,
            cumulative_reward=round(self.cumulative_reward, 4),
            done=self.done,
            available_tools=AVAILABLE_TOOLS,
            tool_results=self.tool_results,
        )

    # ─────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────

    def _record_triage(self, email_id: str, record: Dict[str, Any]) -> None:
        """Move an email from inbox to triaged list."""
        self.triaged_actions.append(record)
        # Remove from agent inbox
        self.inbox = [e for e in self.inbox if e["id"] != email_id]
        # Add to triaged visible list
        email_data = next((e for e in (self._email_map.get(email_id, None),) if e), None)
        self.triaged.append({
            "email_id": email_id,
            "subject": email_data.subject if email_data else "?",
            "priority": record["priority"],
            "category": record["category"],
            "route_to": record["route_to"],
        })

    def _check_done(self) -> bool:
        """Check if episode should end."""
        # Done if all emails triaged OR step limit reached
        if len(self.inbox) == 0:
            self.done = True
        if self.step_count >= self.max_steps:
            self.done = True
        return self.done

    def _build_observation(
        self,
        message: str = "",
        last_tool_result: Optional[Dict[str, Any]] = None,
    ) -> Observation:
        """Build current observation for the agent."""
        return Observation(
            inbox=list(self.inbox),
            triaged=list(self.triaged),
            step_count=self.step_count,
            task_id=self.task_id,
            max_steps=self.max_steps,
            available_tools=AVAILABLE_TOOLS,
            message=message,
        )
