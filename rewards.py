"""
rewards.py — Clamped reward shaping (0.0 to 1.0 only)
Strictly adheres to non-negative requirement for validator.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from models import EmailMessage, TriageAction

# ── Reward constants (All non-negative) ───────────────────────────────────────
R_CORRECT_CATEGORY = 0.20
R_CORRECT_PRIORITY = 0.15
R_CORRECT_ROUTE = 0.15
R_PHISHING_DETECTED = 0.30
R_SECURITY_ESCALATED = 0.35
R_BEC_DETECTED = 0.20
R_TOOL_CORRECT = 0.10
R_TOOL_UNNECESSARY = 0.0     # Clamped penalty
R_CRITICAL_MISSED = 0.0      # Clamped penalty
R_PHISHING_MISSED = 0.0      # Clamped penalty
R_FALSE_POSITIVE_EXEC = 0.0  # Clamped penalty
R_CREDENTIAL_PHISHING_MISSED = 0.0 # Clamped penalty
R_EPISODE_COMPLETE = 0.10
R_WRONG_ESCALATION = 0.0     # Clamped penalty
R_ESCALATION_NO_REASON = 0.0 # Clamped penalty

PRIORITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "spam": 0}

def _priority_close(predicted: Optional[str], true: Optional[str]) -> float:
    if predicted == true: return 1.0
    p, t = PRIORITY_ORDER.get(predicted, -1), PRIORITY_ORDER.get(true, -1)
    if p < 0 or t < 0: return 0.0
    return 0.5 if abs(p - t) == 1 else 0.0

def calculate_triage_reward(action: TriageAction, email: EmailMessage, step: int, task_id: str) -> Tuple[float, List[str]]:
    reward = 0.0
    signals = []
    true_priority, true_category = getattr(email, '_true_priority', None), getattr(email, '_true_category', None)
    true_route, is_phishing = getattr(email, '_true_route', None), getattr(email, '_is_phishing', False)

    if action.category == true_category:
        reward += R_CORRECT_CATEGORY
        signals.append(f"+{R_CORRECT_CATEGORY:.2f} correct category")
        if action.category == "phishing" and is_phishing:
            reward += R_PHISHING_DETECTED
            signals.append(f"+{R_PHISHING_DETECTED:.2f} phishing bonus")
    
    if action.priority == true_priority:
        reward += R_CORRECT_PRIORITY
        signals.append(f"+{R_CORRECT_PRIORITY:.2f} correct priority")
    elif _priority_close(action.priority, true_priority) > 0:
        reward += R_CORRECT_PRIORITY * 0.5
        signals.append(f"+{R_CORRECT_PRIORITY*0.5:.2f} partial priority")

    if action.route_to == true_route:
        reward += R_CORRECT_ROUTE
        signals.append(f"+{R_CORRECT_ROUTE:.2f} correct route")

    return max(0.0, min(1.0, reward)), signals

def calculate_escalation_reward(action: TriageAction, email: Optional[EmailMessage], task_id: str) -> Tuple[float, List[str]]:
    reward = 0.0
    signals = []
    if email:
        true_cat = getattr(email, '_true_category', None)
        if true_cat in ("security_incident", "phishing") and action.escalation_target == "security_team":
            reward = R_SECURITY_ESCALATED
            signals.append(f"+{R_SECURITY_ESCALATED:.2f} correct escalation")
    return max(0.0, min(1.0, reward)), signals

def calculate_tool_reward(tool_name: str, tool_result: Dict[str, Any], task_id: str, count: int, expected: List[str]) -> Tuple[float, List[str]]:
    reward = R_TOOL_CORRECT if tool_name in expected else 0.0
    return max(0.0, min(1.0, reward)), [f"+{reward:.2f} tool reward"]

def calculate_episode_completion_reward(step: int, max_steps: int, all_triaged: bool, task_id: str) -> Tuple[float, List[str]]:
    bonus = R_EPISODE_COMPLETE if all_triaged else 0.0
    return bonus, [f"+{bonus:.2f} completion"]

class RewardNormaliser:
    def __init__(self, n_emails: int, task_id: str):
        self.cumulative = 0.0
        self.n = max(1, n_emails)

    def add(self, raw_reward: float, signals: List[str], step: int) -> None:
        self.cumulative += raw_reward

    def normalised(self) -> float:
        # Simple normalization to [0.0, 1.0] based on email count
        max_possible = self.n * 0.8  # Approx max per email
        return max(0.0, min(1.0, self.cumulative / max_possible if max_possible > 0 else 0.0))
