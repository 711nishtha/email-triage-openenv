"""
rewards.py — Dense reward shaping for the Enterprise Email Triage OpenEnv.

Provides per-action, per-step reward signals to guide the agent with
frequent feedback rather than a single terminal score.

Reward table (per action):
  Correct category          +0.15
  Correct priority          +0.10
  Correct route             +0.10
  Phishing correctly flagged +0.25
  Security incident escalated +0.30
  BEC pattern flagged       +0.20
  Tool used correctly       +0.05
  Tool used unnecessarily   -0.05
  Critical email missed     -0.40
  Phishing not flagged      -0.30
  False positive (legit exec) -0.20
  Credentials not flagged   -0.50
  Episode completed in budget +0.10
  Wrong escalation target   -0.15

All rewards are summed then normalised to [-1.0, 1.0] at episode end.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from models import EmailMessage, TriageAction


# ── Reward constants ─────────────────────────────────────────────────────────

R_CORRECT_CATEGORY = 0.15
R_CORRECT_PRIORITY = 0.10
R_CORRECT_ROUTE = 0.10
R_PHISHING_DETECTED = 0.25
R_SECURITY_ESCALATED = 0.30
R_BEC_DETECTED = 0.20
R_TOOL_CORRECT = 0.05
R_TOOL_UNNECESSARY = -0.05
R_CRITICAL_MISSED = -0.40
R_PHISHING_MISSED = -0.30
R_FALSE_POSITIVE_EXEC = -0.20
R_CREDENTIAL_PHISHING_MISSED = -0.50
R_EPISODE_COMPLETE = 0.10
R_WRONG_ESCALATION = -0.15
R_ESCALATION_NO_REASON = -0.05


# ── Priority adjacency map (for partial credit) ──────────────────────────────

PRIORITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "spam": 0}


def _priority_close(predicted: Optional[str], true: Optional[str]) -> float:
    """Return 1.0 for exact, 0.5 for adjacent, 0.0 for far/wrong."""
    if predicted is None or true is None:
        return 0.0
    if predicted == true:
        return 1.0
    p = PRIORITY_ORDER.get(predicted, -1)
    t = PRIORITY_ORDER.get(true, -1)
    if p < 0 or t < 0:
        return 0.0
    return 0.5 if abs(p - t) == 1 else 0.0


# ── Per-action reward calculator ─────────────────────────────────────────────

def calculate_triage_reward(
    action: TriageAction,
    email: EmailMessage,
    step: int,
    task_id: str,
) -> Tuple[float, List[str]]:
    """
    Calculate the immediate reward for a triage action on a specific email.

    Returns:
      (reward: float, reward_signals: List[str]) — signals are logged for transparency.
    """
    reward = 0.0
    signals: List[str] = []

    true_priority = getattr(email, '_true_priority', None)
    true_category = getattr(email, '_true_category', None)
    true_route = getattr(email, '_true_route', None)
    is_phishing = getattr(email, '_is_phishing', False)
    is_bec = getattr(email, '_is_bec', False)

    # ── Category reward ──────────────────────────────────────────────────────
    if action.category == true_category:
        # Special bonus for phishing detection
        if action.category == "phishing" and is_phishing:
            reward += R_PHISHING_DETECTED
            signals.append(f"+{R_PHISHING_DETECTED:.2f} phishing correctly detected")
            if is_bec:
                reward += R_BEC_DETECTED
                signals.append(f"+{R_BEC_DETECTED:.2f} BEC pattern identified")
        elif action.category == "security_incident":
            reward += R_CORRECT_CATEGORY
            signals.append(f"+{R_CORRECT_CATEGORY:.2f} correct category (security_incident)")
        else:
            reward += R_CORRECT_CATEGORY
            signals.append(f"+{R_CORRECT_CATEGORY:.2f} correct category ({action.category})")
    else:
        # Penalty for missing phishing
        if is_phishing and action.category != "phishing":
            reward += R_PHISHING_MISSED
            signals.append(
                f"{R_PHISHING_MISSED:.2f} phishing email NOT flagged "
                f"(predicted: {action.category})"
            )
            # Extra penalty if it was credential-harvesting
            body_lower = email.body.lower()
            if "password" in body_lower or "credential" in body_lower:
                reward += R_CREDENTIAL_PHISHING_MISSED
                signals.append(
                    f"{R_CREDENTIAL_PHISHING_MISSED:.2f} credential-harvesting phishing missed!"
                )
        # Penalty for false positive: flagging a legitimate exec email as phishing
        elif not is_phishing and action.category == "phishing":
            if true_category in ("executive_request", "security_incident", "hr_matter"):
                reward += R_FALSE_POSITIVE_EXEC
                signals.append(
                    f"{R_FALSE_POSITIVE_EXEC:.2f} false positive — legitimate "
                    f"{true_category} flagged as phishing"
                )

    # ── Priority reward ──────────────────────────────────────────────────────
    if action.priority is not None and true_priority is not None:
        p_score = _priority_close(action.priority, true_priority)
        priority_reward = R_CORRECT_PRIORITY * p_score

        if p_score == 0.0 and true_priority == "critical":
            # Penalty for deprioritising a critical email
            priority_reward += R_CRITICAL_MISSED
            signals.append(
                f"{R_CRITICAL_MISSED:.2f} critical email assigned priority '{action.priority}' — MISSED!"
            )
        elif p_score > 0:
            reward += priority_reward
            suffix = "(exact)" if p_score == 1.0 else "(adjacent, partial)"
            signals.append(f"+{priority_reward:.2f} priority {suffix}")

    # ── Route reward ─────────────────────────────────────────────────────────
    if action.route_to is not None and true_route is not None:
        if action.route_to == true_route:
            reward += R_CORRECT_ROUTE
            signals.append(f"+{R_CORRECT_ROUTE:.2f} correct route ({action.route_to})")
        else:
            # Partial credit checked via family in grader, not here
            pass

    # ── Escalation reward ────────────────────────────────────────────────────
    # (handled by calculate_escalation_reward below)

    # Clamp per-action reward to reasonable bounds
    reward = max(-1.0, min(0.8, reward))
    return reward, signals


def calculate_escalation_reward(
    action: TriageAction,
    email: Optional[EmailMessage],
    task_id: str,
) -> Tuple[float, List[str]]:
    """
    Calculate reward for an escalation action.
    """
    reward = 0.0
    signals: List[str] = []

    if email is None:
        return R_WRONG_ESCALATION, ["Escalation for unknown email_id."]

    true_category = getattr(email, '_true_category', None)
    is_phishing = getattr(email, '_is_phishing', False)

    # Valid escalation targets
    valid_targets = {
        "security_incident": "security_team",
        "phishing": "security_team",
    }

    expected_target = valid_targets.get(true_category)

    if expected_target and action.escalation_target == expected_target:
        if true_category == "security_incident":
            reward = R_SECURITY_ESCALATED
            signals.append(f"+{R_SECURITY_ESCALATED:.2f} security incident correctly escalated")
        elif is_phishing:
            reward = R_PHISHING_DETECTED
            signals.append(f"+{R_PHISHING_DETECTED:.2f} phishing correctly escalated to security")
    elif expected_target and action.escalation_target != expected_target:
        reward = R_WRONG_ESCALATION
        signals.append(
            f"{R_WRONG_ESCALATION:.2f} wrong escalation target "
            f"(got {action.escalation_target}, expected {expected_target})"
        )
    elif not expected_target:
        # Escalating something that shouldn't be escalated
        reward = R_WRONG_ESCALATION
        signals.append(
            f"{R_WRONG_ESCALATION:.2f} unnecessary escalation — "
            f"{true_category} does not require escalation"
        )

    if not action.escalation_reason:
        reward += R_ESCALATION_NO_REASON
        signals.append(f"{R_ESCALATION_NO_REASON:.2f} no escalation reason provided")

    return reward, signals


def calculate_tool_reward(
    tool_name: str,
    tool_result: Dict[str, Any],
    task_id: str,
    tool_call_count: int,
    expected_tools: List[str],
) -> Tuple[float, List[str]]:
    """
    Calculate reward for a tool call.

    Rewards appropriate use; penalises unnecessary/excessive calls.
    """
    reward = 0.0
    signals: List[str] = []

    if not tool_result.get("success", False):
        signals.append(f"Tool '{tool_name}' failed — no reward.")
        return 0.0, signals

    if tool_name in expected_tools:
        reward = R_TOOL_CORRECT
        signals.append(f"+{R_TOOL_CORRECT:.2f} appropriate tool use ({tool_name})")
    else:
        reward = R_TOOL_UNNECESSARY
        signals.append(f"{R_TOOL_UNNECESSARY:.2f} unnecessary tool call ({tool_name})")

    # Escalating penalties for excessive tool use
    if tool_call_count > len(expected_tools) + 2:
        extra_penalty = R_TOOL_UNNECESSARY * (tool_call_count - len(expected_tools) - 2)
        reward += extra_penalty
        signals.append(f"{extra_penalty:.2f} excessive tool use penalty (call #{tool_call_count})")

    return reward, signals


def calculate_episode_completion_reward(
    step: int,
    max_steps: int,
    all_triaged: bool,
    task_id: str,
) -> Tuple[float, List[str]]:
    """
    End-of-episode reward for completing within step budget.
    """
    if not all_triaged:
        return 0.0, ["Episode ended without triaging all emails — no completion bonus."]

    # Graduated bonus: more reward for finishing with fewer steps
    efficiency = 1.0 - (step / max_steps)
    bonus = R_EPISODE_COMPLETE * (0.5 + 0.5 * efficiency)
    return bonus, [f"+{bonus:.2f} episode completion bonus (finished at step {step}/{max_steps})"]


# ── Normaliser ───────────────────────────────────────────────────────────────

class RewardNormaliser:
    """
    Tracks cumulative raw reward during an episode and normalises it to [0.0, 1.0]
    for external reporting.

    The normalisation maps [-max_penalty, max_bonus] → [0.0, 1.0].
    """

    # Approximate episode bounds (loose; adjusted by task)
    MAX_BONUS_PER_EMAIL = 0.60   # category + priority + route + phishing + BEC
    MAX_PENALTY_PER_EMAIL = 1.30  # phishing miss + credential miss + critical miss

    def __init__(self, n_emails: int, task_id: str):
        self.n = max(1, n_emails)
        self.task_id = task_id
        self.cumulative = 0.0
        self.steps: List[Dict[str, Any]] = []

    @property
    def max_possible(self) -> float:
        return self.n * self.MAX_BONUS_PER_EMAIL + R_EPISODE_COMPLETE

    @property
    def min_possible(self) -> float:
        return -(self.n * self.MAX_PENALTY_PER_EMAIL)

    def add(self, raw_reward: float, signals: List[str], step: int) -> None:
        self.cumulative += raw_reward
        self.steps.append({
            "step": step,
            "raw_reward": raw_reward,
            "signals": signals,
            "cumulative": self.cumulative,
        })

    def normalised(self) -> float:
        """Return cumulative reward normalised to [0.0, 1.0]."""
        span = self.max_possible - self.min_possible
        if span == 0:
            return 0.5
        normalised = (self.cumulative - self.min_possible) / span
        return max(0.0, min(1.0, normalised))

    def summary(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "cumulative_raw": self.cumulative,
            "normalised_score": self.normalised(),
            "max_possible": self.max_possible,
            "min_possible": self.min_possible,
            "steps": self.steps,
        }
