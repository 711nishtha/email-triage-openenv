"""
environment.py — Core OpenEnv environment for Advanced Enterprise Email Triage.

Implements the OpenEnv interface:
  reset(task_id, seed) → EmailObservation
  step(action)         → (EmailObservation, reward, done, info)
  state()              → EnvironmentState

Thread-safe for single-user FastAPI server use.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from models import (
    EmailMessage,
    EmailObservation,
    EnvironmentState,
    ResetRequest,
    SenderProfile,
    StepResponse,
    TriageAction,
)
from data import make_easy_inbox, make_medium_inbox, make_hard_inbox
from graders import get_grader
from rewards import (
    RewardNormaliser,
    calculate_triage_reward,
    calculate_escalation_reward,
    calculate_tool_reward,
    calculate_episode_completion_reward,
)
from tools import run_tool, AVAILABLE_TOOLS


# ── Task configuration ───────────────────────────────────────────────────────

TASK_CONFIG = {
    "easy": {
        "max_steps": 5,
        "available_tools": [],          # No tools for easy task
        "expected_tools": [],
    },
    "medium": {
        "max_steps": 15,
        "available_tools": AVAILABLE_TOOLS,
        "expected_tools": ["calendar_check", "kb_search"],
    },
    "hard": {
        "max_steps": 25,
        "available_tools": AVAILABLE_TOOLS,
        "expected_tools": ["sender_lookup"],
    },
}


# ── Environment class ─────────────────────────────────────────────────────────

class EmailTriageEnvironment:
    """
    The main OpenEnv-compatible environment for enterprise email triage.

    State is held in-memory per instance. For multi-user scenarios, each
    session should have its own instance (managed by the FastAPI server).
    """

    def __init__(self):
        # Initialise with a null state
        self._task_id: str = "easy"
        self._episode_id: str = ""
        self._step: int = 0
        self._max_steps: int = 5
        self._done: bool = True
        self._emails: List[EmailMessage] = []
        self._thread_history: Dict[str, List[EmailMessage]] = {}
        self._sender_profiles: Dict[str, SenderProfile] = {}
        self._pending_ids: List[str] = []
        self._triaged_ids: List[str] = []
        self._decisions: List[Dict[str, Any]] = []
        self._tool_call_log: List[Dict[str, Any]] = []
        self._warnings: List[str] = []
        self._normaliser: Optional[RewardNormaliser] = None
        self._available_tools: List[str] = []
        self._expected_tools: List[str] = []
        self._grader_breakdown: Dict[str, float] = {}

    # ── reset ────────────────────────────────────────────────────────────────

    def reset(self, request: ResetRequest) -> EmailObservation:
        """
        Reset the environment to a fresh episode.

        Loads new emails, resets all state, and returns the initial observation.
        """
        task_id = request.task_id
        seed = request.seed

        if task_id not in TASK_CONFIG:
            raise ValueError(f"Unknown task_id '{task_id}'. Choose from {list(TASK_CONFIG.keys())}")

        config = TASK_CONFIG[task_id]
        self._task_id = task_id
        self._episode_id = str(uuid.uuid4())
        self._step = 0
        self._max_steps = config["max_steps"]
        self._done = False
        self._decisions = []
        self._tool_call_log = []
        self._warnings = []
        self._triaged_ids = []
        self._grader_breakdown = {}
        self._available_tools = config["available_tools"]
        self._expected_tools = config["expected_tools"]

        # Load synthetic emails for this task
        if task_id == "easy":
            emails, sender_profiles = make_easy_inbox(seed=seed)
            thread_history: Dict[str, List[EmailMessage]] = {}
        elif task_id == "medium":
            emails, thread_history, sender_profiles = make_medium_inbox(seed=seed)
        else:  # hard
            emails, thread_history, sender_profiles = make_hard_inbox(seed=seed)

        self._emails = emails
        self._thread_history = thread_history
        self._sender_profiles = sender_profiles
        self._pending_ids = [e.email_id for e in emails]

        # Initialise reward normaliser
        self._normaliser = RewardNormaliser(n_emails=len(emails), task_id=task_id)

        return self._build_observation()

    # ── step ─────────────────────────────────────────────────────────────────

    def step(self, action: TriageAction) -> StepResponse:
        """
        Process one agent action and advance the environment.

        Returns StepResponse with new observation, reward, done flag, and info.
        """
        if self._done:
            raise RuntimeError(
                "Episode is already done. Call /reset to start a new episode."
            )

        self._step += 1
        reward = 0.0
        info: Dict[str, Any] = {"action_type": action.action_type, "signals": []}

        # ── Handle action types ──────────────────────────────────────────────

        if action.action_type == "triage":
            reward, signals = self._handle_triage(action)
            info["signals"] = signals

        elif action.action_type == "use_tool":
            reward, signals = self._handle_tool(action)
            info["signals"] = signals

        elif action.action_type == "escalate":
            reward, signals = self._handle_escalate(action)
            info["signals"] = signals

        elif action.action_type == "done":
            reward, signals = self._handle_done()
            info["signals"] = signals

        # ── Accumulate reward ────────────────────────────────────────────────
        if self._normaliser:
            self._normaliser.add(reward, info["signals"], self._step)

        # ── Check termination ────────────────────────────────────────────────
        all_triaged = len(self._triaged_ids) >= len(self._emails)

        if all_triaged or self._step >= self._max_steps or action.action_type == "done":
            self._done = True
            # Add completion bonus if deserved
            completion_reward, c_signals = calculate_episode_completion_reward(
                step=self._step,
                max_steps=self._max_steps,
                all_triaged=all_triaged,
                task_id=self._task_id,
            )
            if self._normaliser:
                self._normaliser.add(completion_reward, c_signals, self._step)
            info["signals"].extend(c_signals)

            # Run terminal grader
            self._run_terminal_grader()
            info["grader_breakdown"] = self._grader_breakdown
            info["final_score"] = self._normaliser.normalised() if self._normaliser else 0.0

        obs = self._build_observation()
        return StepResponse(
            observation=obs,
            reward=round(reward, 4),
            done=self._done,
            info=info,
        )

    # ── state ────────────────────────────────────────────────────────────────

    def state(self) -> EnvironmentState:
        """Return full environment state (includes ground-truth labels for evaluation)."""
        return EnvironmentState(
            task_id=self._task_id,
            episode_id=self._episode_id,
            step=self._step,
            max_steps=self._max_steps,
            done=self._done,
            total_reward=self._normaliser.normalised() if self._normaliser else 0.0,
            emails=self._emails,
            agent_decisions=self._decisions,
            tool_call_log=self._tool_call_log,
            pending_email_ids=self._pending_ids,
            triaged_email_ids=self._triaged_ids,
            warnings=self._warnings,
            grader_breakdown=self._grader_breakdown,
        )

    # ── Action handlers ──────────────────────────────────────────────────────

    def _handle_triage(self, action: TriageAction) -> Tuple[float, List[str]]:
        """Process a triage decision for an email."""
        if not action.email_id:
            return 0.0, ["No email_id provided in triage action."]

        # Find the email
        email = self._get_email(action.email_id)
        if email is None:
            return 0.0, [f"email_id '{action.email_id}' not found in inbox."]

        # Prevent double-triaging
        if action.email_id in self._triaged_ids:
            return 0.0, [f"Email '{action.email_id}' already triaged."]

        # Calculate reward
        reward, signals = calculate_triage_reward(
            action=action,
            email=email,
            step=self._step,
            task_id=self._task_id,
        )

        # Log decision
        self._decisions.append({
            "step": self._step,
            "action_type": "triage",
            "email_id": action.email_id,
            "priority": action.priority,
            "category": action.category,
            "route_to": action.route_to,
            "reasoning": action.reasoning,
        })

        # Mark as triaged
        self._triaged_ids.append(action.email_id)
        if action.email_id in self._pending_ids:
            self._pending_ids.remove(action.email_id)

        # Safety warning for missed phishing
        email_is_phishing = getattr(email, '_is_phishing', False)
        if email_is_phishing and action.category != "phishing":
            self._warnings.append(
                f"SAFETY: Phishing email '{email.subject[:40]}' was not flagged at step {self._step}."
            )

        return reward, signals

    def _handle_tool(self, action: TriageAction) -> Tuple[float, List[str]]:
        """Execute a tool call and return reward."""
        if not action.tool_name:
            return 0.0, ["No tool_name provided."]

        if action.tool_name not in AVAILABLE_TOOLS:
            return 0.0, [f"Unknown tool '{action.tool_name}'."]

        if self._available_tools and action.tool_name not in self._available_tools:
            return 0.0, [f"Tool '{action.tool_name}' not available for task '{self._task_id}'."]

        # Run the tool
        result = run_tool(action.tool_name, action.tool_params or {})

        # Log tool call
        self._tool_call_log.append({
            "step": self._step,
            "tool_name": action.tool_name,
            "params": action.tool_params,
            "success": result.get("success"),
        })

        # Calculate tool reward
        reward, signals = calculate_tool_reward(
            tool_name=action.tool_name,
            tool_result=result,
            task_id=self._task_id,
            tool_call_count=len(self._tool_call_log),
            expected_tools=self._expected_tools,
        )

        signals.append(f"Tool result preview: {str(result.get('result', ''))[:100]}")
        return reward, signals

    def _handle_escalate(self, action: TriageAction) -> Tuple[float, List[str]]:
        """Handle a direct escalation action."""
        if not action.email_id:
            return 0.0, ["No email_id provided in escalate action."]

        email = self._get_email(action.email_id)
        reward, signals = calculate_escalation_reward(
            action=action,
            email=email,
            task_id=self._task_id,
        )

        # Log the escalation as a decision
        self._decisions.append({
            "step": self._step,
            "action_type": "escalate",
            "email_id": action.email_id,
            "escalation_target": action.escalation_target,
            "escalation_reason": action.escalation_reason,
            # Treat escalate as triage for scoring purposes
            "priority": "critical",
            "category": "security_incident" if action.escalation_target == "security_team" else "other",
            "route_to": action.escalation_target,
        })

        if action.email_id not in self._triaged_ids:
            self._triaged_ids.append(action.email_id)
            if action.email_id in self._pending_ids:
                self._pending_ids.remove(action.email_id)

        return reward, signals

    def _handle_done(self) -> Tuple[float, List[str]]:
        """Handle agent signalling it is finished."""
        untriaged = len(self._pending_ids)
        if untriaged > 0:
            return 0.0, [
                f"Done action with {untriaged} email(s) still pending. "
                f"No completion bonus."
            ]
        return 0.0, ["Agent signalled done — all emails processed."]

    def _run_terminal_grader(self) -> None:
        """Run the end-of-episode grader and store the breakdown."""
        try:
            grader = get_grader(self._task_id)
            _, breakdown, _ = grader.grade(
                emails=self._emails,
                decisions=self._decisions,
                tool_calls=self._tool_call_log,
                warnings=self._warnings,
            )
            self._grader_breakdown = breakdown
        except Exception as exc:
            self._grader_breakdown = {"error": str(exc)}

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _get_email(self, email_id: str) -> Optional[EmailMessage]:
        """Look up an email by ID from the inbox."""
        return next((e for e in self._emails if e.email_id == email_id), None)

    def _build_observation(self) -> EmailObservation:
        """Construct the observation returned to the agent."""
        # Only show pending emails in inbox (not yet triaged)
        pending_emails = [e for e in self._emails if e.email_id in self._pending_ids]

        return EmailObservation(
            task_id=self._task_id,
            step=self._step,
            max_steps=self._max_steps,
            inbox=pending_emails,
            thread_history=self._thread_history,
            sender_profiles=self._sender_profiles,
            available_tools=self._available_tools,
            current_score=self._normaliser.normalised() if self._normaliser else 0.0,
            warnings=self._warnings,
            done=self._done,
            info={
                "episode_id": self._episode_id,
                "pending_count": len(self._pending_ids),
                "triaged_count": len(self._triaged_ids),
            },
        )
