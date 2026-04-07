"""
graders.py — Deterministic graders with final score clamped to (0,1).
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from models import EmailWithGroundTruth

WEIGHT_CATEGORY = 0.4
WEIGHT_PRIORITY = 0.3
WEIGHT_ROUTE = 0.3
PENALTY_MISSED_URGENT = 0.3
PENALTY_PHISHING_MISS = 0.4

def grade_single_action(
    action_category: str,
    action_priority: str,
    action_route: str,
    email: EmailWithGroundTruth,
) -> Tuple[float, Dict[str, Any]]:
    score = 0.0
    breakdown = {
        "email_id": email.id,
        "category_correct": False,
        "priority_correct": False,
        "route_correct": False,
        "category_score": 0.0,
        "priority_score": 0.0,
        "route_score": 0.0,
        "total_score": 0.0,
        "expected_category": email.expected_category,
        "expected_priority": email.expected_priority,
        "expected_route": email.expected_route,
        "got_category": action_category,
        "got_priority": action_priority,
        "got_route": action_route,
    }
    # Category
    if action_category == email.expected_category:
        score += WEIGHT_CATEGORY
        breakdown["category_correct"] = True
        breakdown["category_score"] = WEIGHT_CATEGORY
    else:
        partial = _partial_category_credit(action_category, email.expected_category)
        score += partial
        breakdown["category_score"] = partial
    # Priority
    if action_priority == email.expected_priority:
        score += WEIGHT_PRIORITY
        breakdown["priority_correct"] = True
        breakdown["priority_score"] = WEIGHT_PRIORITY
    else:
        partial = _partial_priority_credit(action_priority, email.expected_priority)
        score += partial
        breakdown["priority_score"] = partial
    # Routing
    if action_route == email.expected_route:
        score += WEIGHT_ROUTE
        breakdown["route_correct"] = True
        breakdown["route_score"] = WEIGHT_ROUTE
    else:
        partial = _partial_route_credit(action_route, email.expected_route)
        score += partial
        breakdown["route_score"] = partial
    breakdown["total_score"] = round(min(1.0, max(0.0, score)), 4)
    return breakdown["total_score"], breakdown

def _partial_category_credit(got: str, expected: str) -> float:
    adjacency = {
        frozenset({"urgent_business", "internal_task"}): 0.1,
        frozenset({"spam", "marketing"}): 0.2,
        frozenset({"legal", "finance"}): 0.1,
        frozenset({"phishing", "spam"}): 0.0,
        frozenset({"it_support", "internal_task"}): 0.1,
        frozenset({"hr", "internal_task"}): 0.1,
    }
    return adjacency.get(frozenset({got, expected}), 0.0)

def _partial_priority_credit(got: str, expected: str) -> float:
    order = ["urgent", "high", "medium", "low"]
    try:
        gi, ei = order.index(got), order.index(expected)
        if gi == ei:
            return WEIGHT_PRIORITY
        elif abs(gi - ei) == 1:
            return WEIGHT_PRIORITY * 0.5
        return 0.0
    except ValueError:
        return 0.0

def _partial_route_credit(got: str, expected: str) -> float:
    groups = {
        frozenset({"executive", "manager"}): 0.15,
        frozenset({"it", "security"}): 0.1,
        frozenset({"archive", "trash"}): 0.15,
        frozenset({"finance", "manager"}): 0.1,
        frozenset({"hr", "manager"}): 0.1,
    }
    return groups.get(frozenset({got, expected}), 0.0)

def grade_episode(
    task_emails: List[EmailWithGroundTruth],
    triaged_actions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    action_map = {a["email_id"]: a for a in triaged_actions}
    per_email_scores = []
    total_raw = 0.0
    penalties = []
    total_penalty = 0.0
    for email in task_emails:
        action = action_map.get(email.id)
        if action is None:
            if email.expected_priority == "urgent":
                total_penalty += PENALTY_MISSED_URGENT
                penalties.append({"email_id": email.id, "type": "missed_urgent", "penalty": -PENALTY_MISSED_URGENT})
            if email.expected_category == "phishing":
                total_penalty += PENALTY_PHISHING_MISS
                penalties.append({"email_id": email.id, "type": "missed_phishing", "penalty": -PENALTY_PHISHING_MISS})
            per_email_scores.append({"email_id": email.id, "total_score": 0.0})
        else:
            score, breakdown = grade_single_action(
                action.get("category", ""),
                action.get("priority", ""),
                action.get("route_to", ""),
                email
            )
            if email.expected_category == "phishing" and action.get("category") != "phishing":
                total_penalty += PENALTY_PHISHING_MISS
                penalties.append({"email_id": email.id, "type": "phishing_misclassified", "penalty": -PENALTY_PHISHING_MISS})
            total_raw += score
            per_email_scores.append(breakdown)
    num = len(task_emails)
    avg = total_raw / num if num > 0 else 0.0
    penalty_rate = total_penalty / num if num > 0 else 0.0
    final = avg - penalty_rate
    # CRITICAL: clamp to (0,1) – never 0.0 or 1.0
    if final <= 0.0:
        final = 0.001
    elif final >= 1.0:
        final = 0.999
    final = round(final, 4)
    return {
        "episode_score": final,
        "raw_score": round(avg, 4),
        "penalty_total": round(total_penalty, 4),
        "per_email_scores": per_email_scores,
        "penalties": penalties,
        "num_triaged": len(triaged_actions),
        "num_emails": num,
        "coverage": round(len(triaged_actions) / num, 4) if num > 0 else 0.0,
    }

def compute_step_reward(
    action_type: str,
    email: Optional[EmailWithGroundTruth],
    action_category: Optional[str],
    action_priority: Optional[str],
    action_route: Optional[str],
) -> float:
    if action_type != "triage" or email is None:
        return 0.001  # never zero
    score, _ = grade_single_action(
        action_category or "",
        action_priority or "",
        action_route or "",
        email
    )
    if score <= 0.0:
        return 0.001
    if score >= 1.0:
        return 0.999
    return score
