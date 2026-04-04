"""
rewards.py — Dense reward computation for the email triage environment.

Aggregates step-level and episode-level rewards.
All values are clamped to [0.0, 1.0].
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from models import EmailWithGroundTruth
from graders import (
    compute_step_reward,
    grade_episode,
    PENALTY_MISSED_URGENT,
    PENALTY_PHISHING_MISS,
)


def compute_reward(
    action_type: str,
    email: Optional[EmailWithGroundTruth],
    action_category: Optional[str],
    action_priority: Optional[str],
    action_route: Optional[str],
    is_done: bool,
    task_emails: List[EmailWithGroundTruth],
    triaged_actions: List[Dict[str, Any]],
    step_count: int,
    max_steps: int,
) -> Dict[str, Any]:
    """
    Compute the reward for a step.

    For non-final steps: immediate partial reward based on action quality.
    For the final step (done=True): episode-level summary reward.

    Returns:
        {
            "reward": float,          # Clamped [0.0, 1.0]
            "step_reward": float,     # Immediate reward this step
            "is_final": bool,
            "episode_summary": dict,  # Only if is_final
        }
    """
    step_reward = compute_step_reward(
        action_type=action_type,
        email=email,
        action_category=action_category,
        action_priority=action_priority,
        action_route=action_route,
    )

    result: Dict[str, Any] = {
        "reward": round(max(0.0, min(1.0, step_reward)), 4),
        "step_reward": step_reward,
        "is_final": is_done,
        "episode_summary": None,
    }

    if is_done:
        summary = grade_episode(
            task_emails=task_emails,
            triaged_actions=triaged_actions,
        )
        result["episode_summary"] = summary
        # Override reward with final episode score when done
        result["reward"] = summary["episode_score"]

    return result


def compute_completion_bonus(
    triaged_count: int,
    total_emails: int,
    step_count: int,
    max_steps: int,
) -> float:
    """
    Bonus for completing all emails efficiently.
    Returns a small bonus (up to 0.05) for triaging all emails within step budget.
    """
    if triaged_count < total_emails:
        return 0.0
    efficiency = 1.0 - (step_count / max_steps)
    return round(min(0.05, efficiency * 0.05), 4)


def explain_reward(reward_info: Dict[str, Any]) -> str:
    """Return a human-readable explanation of the reward."""
    lines = []
    lines.append(f"Step Reward: {reward_info['step_reward']:.4f}")
    lines.append(f"Final Reward: {reward_info['reward']:.4f}")

    if reward_info.get("episode_summary"):
        s = reward_info["episode_summary"]
        lines.append(f"Episode Score: {s['episode_score']:.4f}")
        lines.append(f"  Raw Score: {s['raw_score']:.4f}")
        lines.append(f"  Penalty Total: {s['penalty_total']:.4f}")
        lines.append(f"  Emails Triaged: {s['num_triaged']}/{s['num_emails']}")
        lines.append(f"  Coverage: {s['coverage']:.2%}")

        if s["penalties"]:
            lines.append("  Penalties Applied:")
            for p in s["penalties"]:
                lines.append(f"    - {p['reason']} ({p['penalty']:.2f})")

    return "\n".join(lines)
