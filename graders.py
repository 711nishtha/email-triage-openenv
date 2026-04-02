"""
graders.py — Deterministic task graders for all three difficulty levels.

Each grader receives the full episode state and returns:
  - score: float in [0.0, 1.0]
  - breakdown: Dict[str, float] with per-component scores
  - feedback: List[str] with human-readable evaluation notes

Grading is deterministic and programmatic — no LLM judge used.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from models import EmailMessage, TriageAction


# ── Priority adjacency for partial credit ────────────────────────────────────

PRIORITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1, "spam": 0}

def _priority_score(predicted: Optional[str], true: Optional[str]) -> float:
    """
    Return score for priority prediction.
    Exact match: 1.0. Adjacent: 0.5. Two away: 0.25. Further: 0.0.
    """
    if predicted is None or true is None:
        return 0.0
    if predicted == true:
        return 1.0
    p_pred = PRIORITY_ORDER.get(predicted, -1)
    p_true = PRIORITY_ORDER.get(true, -1)
    if p_pred < 0 or p_true < 0:
        return 0.0
    distance = abs(p_pred - p_true)
    if distance == 1:
        return 0.5
    if distance == 2:
        return 0.25
    return 0.0


def _category_score(predicted: Optional[str], true: Optional[str]) -> float:
    """Exact match required for category. 1.0 or 0.0."""
    if predicted is None or true is None:
        return 0.0
    return 1.0 if predicted == true else 0.0


def _route_score(predicted: Optional[str], true: Optional[str]) -> float:
    """
    Routing score with partial credit for correct team family.
    E.g., routing to 'ceo_office' when 'executive_assistant' was expected
    is partially correct (same executive family).
    """
    if predicted is None or true is None:
        return 0.0
    if predicted == true:
        return 1.0

    # Define routing families for partial credit
    FAMILIES = {
        "executive": {"ceo_office", "cfo_office", "executive_assistant"},
        "security": {"security_team"},
        "legal": {"legal_team"},
        "finance": {"finance_team", "cfo_office"},
        "engineering": {"engineering_team", "it_helpdesk"},
        "hr": {"hr_team"},
        "archive": {"archive", "spam_folder"},
        "sales": {"sales_team", "customer_success"},
    }

    pred_family = None
    true_family = None
    for family, members in FAMILIES.items():
        if predicted in members:
            pred_family = family
        if true in members:
            true_family = family

    if pred_family and true_family and pred_family == true_family:
        return 0.5  # Same family, partial credit

    return 0.0


# ── Easy task grader ─────────────────────────────────────────────────────────

class EasyTaskGrader:
    """
    Grader for Task 1: Morning Inbox Clear.
    Single email — priority (30%), category (40%), route (30%).
    """

    WEIGHTS = {"priority": 0.30, "category": 0.40, "route": 0.30}

    def grade(
        self,
        emails: List[EmailMessage],
        decisions: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        warnings: List[str],
    ) -> Tuple[float, Dict[str, float], List[str]]:
        """
        Returns (score, breakdown, feedback_messages).
        Score is in [0.0, 1.0].
        """
        feedback: List[str] = []
        breakdown: Dict[str, float] = {}

        if not emails:
            return 0.0, {"error": 0.0}, ["No emails to grade."]

        email = emails[0]  # Single email task

        # Find the triage decision for this email
        decision = next(
            (d for d in decisions if d.get("email_id") == email.email_id), None
        )

        if decision is None:
            feedback.append(f"No triage decision found for email {email.email_id}.")
            return 0.0, {"priority": 0.0, "category": 0.0, "route": 0.0}, feedback

        # Retrieve hidden ground-truth
        true_priority = getattr(email, '_true_priority', None)
        true_category = getattr(email, '_true_category', None)
        true_route = getattr(email, '_true_route', None)

        # Score each component
        p_score = _priority_score(decision.get("priority"), true_priority)
        c_score = _category_score(decision.get("category"), true_category)
        r_score = _route_score(decision.get("route_to"), true_route)

        breakdown["priority"] = p_score
        breakdown["category"] = c_score
        breakdown["route"] = r_score

        # Feedback
        _add_comparison_feedback(
            feedback, "Priority", decision.get("priority"), true_priority, p_score
        )
        _add_comparison_feedback(
            feedback, "Category", decision.get("category"), true_category, c_score
        )
        _add_comparison_feedback(
            feedback, "Route", decision.get("route_to"), true_route, r_score
        )

        # Unnecessary tool use penalty for easy task
        tool_penalty = 0.0
        if tool_calls:
            tool_penalty = min(0.10, len(tool_calls) * 0.05)
            breakdown["tool_efficiency_penalty"] = -tool_penalty
            feedback.append(
                f"Tool use penalty: -{tool_penalty:.2f} ({len(tool_calls)} unnecessary tool call(s) on easy task)."
            )

        # Weighted final score
        raw = (
            p_score * self.WEIGHTS["priority"]
            + c_score * self.WEIGHTS["category"]
            + r_score * self.WEIGHTS["route"]
        )
        final = max(0.0, min(1.0, raw - tool_penalty))

        breakdown["final"] = final
        feedback.append(f"Final score: {final:.3f}")
        return final, breakdown, feedback


# ── Medium task grader ───────────────────────────────────────────────────────

class MediumTaskGrader:
    """
    Grader for Task 2: Batch Triage with Thread Context.

    Scoring:
      - Per-email: category (25%), priority (25%), route (25%)
      - Tool efficiency (25%): reward appropriate tool use, penalise excess
      - Ordering bonus: +0.10 if critical emails processed before low-priority
    """

    EMAIL_WEIGHT = 0.75    # 75% of score from per-email accuracy
    TOOL_WEIGHT = 0.25     # 25% from tool efficiency

    def grade(
        self,
        emails: List[EmailMessage],
        decisions: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        warnings: List[str],
    ) -> Tuple[float, Dict[str, float], List[str]]:
        feedback: List[str] = []
        breakdown: Dict[str, float] = {}

        if not emails:
            return 0.0, {}, ["No emails to grade."]

        # Grade each email
        per_email_scores = []
        for email in emails:
            decision = next(
                (d for d in decisions if d.get("email_id") == email.email_id), None
            )
            if decision is None:
                per_email_scores.append(0.0)
                feedback.append(f"Email {email.email_id} ('{email.subject[:40]}') — NOT triaged.")
                continue

            true_priority = getattr(email, '_true_priority', None)
            true_category = getattr(email, '_true_category', None)
            true_route = getattr(email, '_true_route', None)

            p = _priority_score(decision.get("priority"), true_priority)
            c = _category_score(decision.get("category"), true_category)
            r = _route_score(decision.get("route_to"), true_route)
            email_score = (p + c + r) / 3.0

            per_email_scores.append(email_score)
            breakdown[f"email_{email.email_id[:6]}_score"] = email_score
            feedback.append(
                f"Email '{email.subject[:40]}': "
                f"priority={p:.2f}, category={c:.2f}, route={r:.2f} → {email_score:.2f}"
            )

        avg_email_score = sum(per_email_scores) / len(per_email_scores) if per_email_scores else 0.0
        breakdown["avg_email_accuracy"] = avg_email_score

        # Tool efficiency scoring
        # Expected tool calls for this task: 1 calendar_check, 1 kb_search = 2 total
        expected_tool_calls = 2
        actual_tool_calls = len(tool_calls)
        correct_tools_used = sum(
            1 for t in tool_calls
            if t.get("tool_name") in {"calendar_check", "kb_search"}
        )

        if actual_tool_calls == 0:
            tool_score = 0.4  # Penalise for not using needed tools
            feedback.append("Tool efficiency: 0.40 — needed calendar_check and kb_search but none used.")
        elif actual_tool_calls <= expected_tool_calls + 1:
            tool_score = min(1.0, correct_tools_used / expected_tool_calls)
            feedback.append(
                f"Tool efficiency: {tool_score:.2f} — "
                f"{correct_tools_used}/{expected_tool_calls} expected tools used correctly."
            )
        else:
            excess = actual_tool_calls - expected_tool_calls
            tool_score = max(0.0, 1.0 - (excess * 0.15))
            feedback.append(
                f"Tool efficiency: {tool_score:.2f} — "
                f"{excess} excess tool call(s) incurred penalty."
            )

        breakdown["tool_efficiency"] = tool_score

        # Ordering bonus: check if critical emails were triaged before low-priority ones
        ordering_bonus = _check_ordering_bonus(emails, decisions)
        breakdown["ordering_bonus"] = ordering_bonus
        if ordering_bonus > 0:
            feedback.append(f"Ordering bonus: +{ordering_bonus:.2f} — critical emails processed first.")

        # Final score
        raw = (
            avg_email_score * self.EMAIL_WEIGHT
            + tool_score * self.TOOL_WEIGHT
            + ordering_bonus
        )
        final = max(0.0, min(1.0, raw))
        breakdown["final"] = final
        feedback.append(f"Final score: {final:.3f}")
        return final, breakdown, feedback


def _check_ordering_bonus(
    emails: List[EmailMessage],
    decisions: List[Dict[str, Any]],
) -> float:
    """Return 0.10 bonus if all critical emails were triaged before any low/spam emails."""
    # Build decision order map (email_id → decision step index)
    decision_order = {d["email_id"]: i for i, d in enumerate(decisions)}

    critical_steps = []
    low_steps = []

    for email in emails:
        eid = email.email_id
        if eid not in decision_order:
            continue
        true_priority = getattr(email, '_true_priority', None)
        if true_priority == "critical":
            critical_steps.append(decision_order[eid])
        elif true_priority in ("low", "spam"):
            low_steps.append(decision_order[eid])

    if not critical_steps or not low_steps:
        return 0.0  # No ordering comparison possible

    return 0.10 if max(critical_steps) < min(low_steps) else 0.0


# ── Hard task grader ─────────────────────────────────────────────────────────

class HardTaskGrader:
    """
    Grader for Task 3: Threat Inbox — Phishing, Urgency, Ambiguity.

    Scoring:
      - Phishing detection F1 score (40%)
      - Security incident escalation (20%)
      - Legitimate email accuracy (30%)
      - Safety compliance (10%, with severe penalties for misses)

    Penalties:
      - Credential-harvesting phishing not flagged: -0.30
      - BEC wire-transfer not flagged: -0.25
      - Critical security incident not escalated: -0.25
      - False positive on legitimate exec email: -0.10 per occurrence
    """

    WEIGHTS = {
        "phishing_f1": 0.40,
        "security_escalation": 0.20,
        "legitimate_accuracy": 0.30,
        "safety_compliance": 0.10,
    }

    def grade(
        self,
        emails: List[EmailMessage],
        decisions: List[Dict[str, Any]],
        tool_calls: List[Dict[str, Any]],
        warnings: List[str],
    ) -> Tuple[float, Dict[str, float], List[str]]:
        feedback: List[str] = []
        breakdown: Dict[str, float] = {}
        penalty = 0.0

        # ── Classify ground truth ────────────────────────────────────────────
        phishing_ids = {
            e.email_id for e in emails if getattr(e, '_is_phishing', False)
        }
        security_incident_ids = {
            e.email_id for e in emails
            if getattr(e, '_true_category', '') == 'security_incident'
        }
        bec_ids = {
            e.email_id for e in emails if getattr(e, '_is_bec', False)
        }
        legitimate_emails = [
            e for e in emails if not getattr(e, '_is_phishing', False)
        ]

        # ── Phishing detection F1 ────────────────────────────────────────────
        predicted_phishing = set()
        for decision in decisions:
            eid = decision.get("email_id")
            category = decision.get("category")
            priority = decision.get("priority")
            escalation_target = decision.get("escalation_target")
            # Agent detects phishing by: category=phishing OR route=security_team OR escalate
            if category == "phishing" or (
                priority == "spam" and decision.get("route_to") == "security_team"
            ) or escalation_target == "security_team":
                predicted_phishing.add(eid)

        tp = len(predicted_phishing & phishing_ids)
        fp = len(predicted_phishing - phishing_ids)
        fn = len(phishing_ids - predicted_phishing)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        breakdown["phishing_precision"] = precision
        breakdown["phishing_recall"] = recall
        breakdown["phishing_f1"] = f1
        feedback.append(
            f"Phishing detection — P={precision:.2f}, R={recall:.2f}, F1={f1:.2f} "
            f"(TP={tp}, FP={fp}, FN={fn})"
        )

        # Check BEC specifically (harder to detect than obvious phishing)
        bec_detected = len(predicted_phishing & bec_ids)
        bec_bonus = 0.0
        if bec_ids:
            bec_bonus = 0.10 * (bec_detected / len(bec_ids))
            if bec_detected > 0:
                feedback.append(f"BEC detection bonus: +{bec_bonus:.2f} ({bec_detected}/{len(bec_ids)} BEC detected)")

        # ── Security incident escalation ─────────────────────────────────────
        escalated_ids = {
            d.get("email_id") for d in decisions
            if d.get("action_type") == "escalate"
            or d.get("escalation_target") == "security_team"
            or (d.get("category") == "security_incident" and d.get("route_to") == "security_team")
        }
        sec_escalated = len(escalated_ids & security_incident_ids)
        sec_score = sec_escalated / len(security_incident_ids) if security_incident_ids else 1.0
        breakdown["security_escalation"] = sec_score

        if sec_score < 1.0:
            penalty += 0.25
            feedback.append(
                f"Security escalation: {sec_score:.2f} — MISSED critical security incident! "
                f"Penalty: -0.25"
            )
        else:
            feedback.append(f"Security escalation: 1.00 — all security incidents correctly escalated.")

        # ── Legitimate email accuracy ─────────────────────────────────────────
        legit_scores = []
        for email in legitimate_emails:
            eid = email.email_id
            decision = next((d for d in decisions if d.get("email_id") == eid), None)
            if decision is None:
                legit_scores.append(0.0)
                continue

            true_priority = getattr(email, '_true_priority', None)
            true_category = getattr(email, '_true_category', None)
            true_route = getattr(email, '_true_route', None)

            p = _priority_score(decision.get("priority"), true_priority)
            c = _category_score(decision.get("category"), true_category)
            r = _route_score(decision.get("route_to"), true_route)
            score = (p + c + r) / 3.0
            legit_scores.append(score)

            # False positive check: legitimate exec email classified as phishing
            if true_category in ("executive_request", "security_incident"):
                if decision.get("category") == "phishing":
                    penalty += 0.10
                    feedback.append(
                        f"False positive: legitimate exec email '{email.subject[:40]}' "
                        f"incorrectly flagged as phishing. Penalty: -0.10"
                    )

        legit_accuracy = sum(legit_scores) / len(legit_scores) if legit_scores else 0.0
        breakdown["legitimate_accuracy"] = legit_accuracy
        feedback.append(f"Legitimate email accuracy: {legit_accuracy:.2f}")

        # ── Safety compliance ─────────────────────────────────────────────────
        safety_score = 1.0

        # Credential phishing not flagged = severe penalty
        credential_phishing_ids = {
            e.email_id for e in emails
            if getattr(e, '_is_phishing', False)
            and ("password" in e.body.lower() or "credential" in e.body.lower())
        }
        undetected_credential = credential_phishing_ids - predicted_phishing
        if undetected_credential:
            penalty += 0.30
            safety_score -= 0.50
            feedback.append(
                f"CRITICAL SAFETY MISS: {len(undetected_credential)} credential-harvesting "
                f"phishing email(s) not detected! Penalty: -0.30"
            )

        # BEC not flagged = significant penalty
        undetected_bec = bec_ids - predicted_phishing
        if undetected_bec:
            penalty += 0.25
            safety_score -= 0.40
            feedback.append(
                f"SAFETY MISS: {len(undetected_bec)} BEC email(s) not detected! Penalty: -0.25"
            )

        safety_score = max(0.0, safety_score)
        breakdown["safety_compliance"] = safety_score

        # ── Weighted final score ─────────────────────────────────────────────
        raw = (
            f1 * self.WEIGHTS["phishing_f1"]
            + sec_score * self.WEIGHTS["security_escalation"]
            + legit_accuracy * self.WEIGHTS["legitimate_accuracy"]
            + safety_score * self.WEIGHTS["safety_compliance"]
            + bec_bonus
        )
        final = max(0.0, min(1.0, raw - penalty))
        breakdown["penalty_total"] = -penalty
        breakdown["bec_bonus"] = bec_bonus
        breakdown["final"] = final
        feedback.append(f"Raw: {raw:.3f}, Penalties: -{penalty:.3f}, Final: {final:.3f}")
        return final, breakdown, feedback


# ── Convenience dispatcher ───────────────────────────────────────────────────

def get_grader(task_id: str):
    """Return the appropriate grader for a task."""
    graders = {
        "easy": EasyTaskGrader(),
        "medium": MediumTaskGrader(),
        "hard": HardTaskGrader(),
    }
    if task_id not in graders:
        raise ValueError(f"Unknown task_id '{task_id}'. Choose from: {list(graders.keys())}")
    return graders[task_id]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _add_comparison_feedback(
    feedback: List[str],
    field: str,
    predicted: Optional[str],
    true: Optional[str],
    score: float,
) -> None:
    """Append a human-readable feedback line comparing predicted vs true."""
    predicted_str = predicted or "(none)"
    true_str = true or "(none)"
    status = "✓" if score >= 1.0 else ("~" if score >= 0.5 else "✗")
    feedback.append(
        f"  {status} {field}: predicted='{predicted_str}', expected='{true_str}' → {score:.2f}"
    )
