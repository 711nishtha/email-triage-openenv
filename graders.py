"""
graders.py — Deterministic graders for scoring agent triage decisions.

Scoring breakdown per email:
  - Correct category:  +0.4
  - Correct priority:  +0.3
  - Correct routing:   +0.3
  Total maximum:        1.0

Penalties applied at episode end:
  - Missed urgent email:       -0.3 per email
  - Phishing misclassified:    -0.4 per email

Final episode score is clamped to [0.001, 0.999] to satisfy validator requirements.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from models import EmailWithGroundTruth


# ─────────────────────────────────────────────
# Per-Email Scoring
# ─────────────────────────────────────────────

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
    """Score a single triage action. Returns (score, breakdown)."""
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
    key = frozenset({got, expected})
    return adjacency.get(key, 0.0)


def _partial_priority_credit(got: str, expected: str) -> float:
    priority_order = ["urgent", "high", "medium", "low"]
    try:
        got_idx = priority_order.index(got)
        exp_idx = priority_order.index(expected)
        distance = abs(got_idx - exp_idx)
        if distance == 0:
            return WEIGHT_PRIORITY
        elif distance == 1:
            return WEIGHT_PRIORITY * 0.5
        else:
            return 0.0
    except ValueError:
        return 0.0


def _partial_route_credit(got: str, expected: str) -> float:
    route_groups = {
        frozenset({"executive", "manager"}): 0.15,
        frozenset({"it", "security"}): 0.1,
        frozenset({"archive", "trash"}): 0.15,
        frozenset({"finance", "manager"}): 0.1,
        frozenset({"hr", "manager"}): 0.1,
    }
    key = frozenset({got, expected})
    return route_groups.get(key, 0.0)


# ─────────────────────────────────────────────
# Episode-Level Grading
# ─────────────────────────────────────────────

def grade_episode(
    task_emails: List[EmailWithGroundTruth],
    triaged_actions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Grade the entire episode. Returns dict with episode_score in [0.001, 0.999]."""
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
                penalties.append({
                    "email_id": email.id,
                    "type": "missed_urgent",
                    "penalty": -PENALTY_MISSED_URGENT,
                    "reason": f"Urgent email '{email.id}' was never triaged",
                })
            if email.expected_category == "phishing":
                total_penalty += PENALTY_PHISHING_MISS
                penalties.append({
                    "email_id": email.id,
                    "type": "missed_phishing",
                    "penalty": -PENALTY_PHISHING_MISS,
                    "reason": f"Phishing email '{email.id}' was never identified",
                })
            per_email_scores.append({"email_id": email.id, "total_score": 0.0, "reason": "Not triaged"})
        else:
            score, breakdown = grade_single_action(
                action_category=action.get("category", ""),
                action_priority=action.get("priority", ""),
                action_route=action.get("route_to", ""),
                email=email,
            )
            if email.expected_category == "phishing" and action.get("category") != "phishing":
                total_penalty += PENALTY_PHISHING_MISS
                penalties.append({
                    "email_id": email.id,
                    "type": "phishing_misclassified",
                    "penalty": -PENALTY_PHISHING_MISS,
                    "reason": f"Phishing email classified as '{action.get('category')}'",
                })
            total_raw += score
            per_email_scores.append(breakdown)

    num_emails = len(task_emails)
    avg_score = total_raw / num_emails if num_emails > 0 else 0.0
    penalty_rate = total_penalty / num_emails if num_emails > 0 else 0.0
    final_score = avg_score - penalty_rate

    # Clamp to [0.001, 0.999] — NEVER allow exact 0.0 or 1.0
    if final_score <= 0.0:
        final_score = 0.001
    elif final_score >= 1.0:
        final_score = 0.999

    final_score = round(final_score, 4)

    return {
        "episode_score": final_score,
        "raw_score": round(avg_score, 4),
        "penalty_total": round(total_penalty, 4),
        "per_email_scores": per_email_scores,
        "penalties": penalties,
        "num_triaged": len(triaged_actions),
        "num_emails": num_emails,
        "coverage": round(len(triaged_actions) / num_emails, 4) if num_emails > 0 else 0.0,
    }


def compute_step_reward(
    action_type: str,
    email: Optional[EmailWithGroundTruth],
    action_category: Optional[str],
    action_priority: Optional[str],
    action_route: Optional[str],
) -> float:
    """Step reward for a single action (clamped to [0.001, 0.999] if non-zero)."""
    if action_type != "triage" or email is None:
        return 0.0
    score, _ = grade_single_action(
        action_category=action_category or "",
        action_priority=action_priority or "",
        action_route=action_route or "",
        email=email,
    )
    # Step reward can be 0.0 for non-triage actions, but for triage actions we also avoid 0.0/1.0
    if score <= 0.0:
        return 0.001
    elif score >= 1.0:
        return 0.999
    return score
