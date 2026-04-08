from __future__ import annotations
from typing import Any, Dict, List, Optional
from models import EmailWithGroundTruth
from graders import compute_step_reward, grade_episode

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
    step_reward = compute_step_reward(
        action_type, email, action_category, action_priority, action_route
    )
    result = {
        "reward": step_reward,
        "step_reward": step_reward,
        "is_final": is_done,
        "episode_summary": None,
    }
    if is_done:
        summary = grade_episode(task_emails, triaged_actions)
        result["episode_summary"] = summary
        result["reward"] = summary["episode_score"]
    return result
