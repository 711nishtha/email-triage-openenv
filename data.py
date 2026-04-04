"""
data.py — Synthetic email generation with ground truth labels.
All data is deterministic (no randomness at grading time).
"""

from __future__ import annotations
from typing import Dict, List
from models import EmailWithGroundTruth


# ─────────────────────────────────────────────
# EASY TASK — 2 emails, clear signals
# ─────────────────────────────────────────────

EASY_EMAILS: List[EmailWithGroundTruth] = [
    EmailWithGroundTruth(
        id="easy_001",
        sender="ceo@acmecorp.com",
        subject="URGENT: Board presentation needed by 3PM TODAY",
        body=(
            "I need the Q3 financial summary slides prepared and sent to the board "
            "immediately. The meeting has been moved up to 3PM today. Please escalate "
            "to the finance team and confirm receipt. This is time-critical."
        ),
        timestamp="2024-01-15T09:00:00Z",
        expected_priority="urgent",
        expected_category="urgent_business",
        expected_route="executive",
    ),
    EmailWithGroundTruth(
        id="easy_002",
        sender="newsletter@deals4u-promo.biz",
        subject="🔥 50% OFF Everything — Limited Time Offer!",
        body=(
            "Don't miss out! Huge savings on all products this week only. "
            "Click here to claim your discount before it expires. "
            "Unsubscribe at the bottom of this email."
        ),
        timestamp="2024-01-15T08:45:00Z",
        expected_priority="low",
        expected_category="marketing",
        expected_route="trash",
    ),
]


# ─────────────────────────────────────────────
# MEDIUM TASK — 6 emails, mixed signals
# ─────────────────────────────────────────────

MEDIUM_EMAILS: List[EmailWithGroundTruth] = [
    EmailWithGroundTruth(
        id="med_001",
        sender="cfo@acmecorp.com",
        subject="Q4 Budget Approval Required — Deadline Friday",
        body=(
            "Hi team, I need final sign-off on the Q4 department budgets by end of day Friday. "
            "Please review the attached spreadsheet and confirm your department's allocation. "
            "Delays will impact procurement timelines."
        ),
        timestamp="2024-01-15T07:30:00Z",
        expected_priority="high",
        expected_category="finance",
        expected_route="finance",
    ),
    EmailWithGroundTruth(
        id="med_002",
        sender="hr@acmecorp.com",
        subject="Annual Performance Review — Schedule Your Session",
        body=(
            "Dear team member, it's time for annual performance reviews. "
            "Please use the calendar link to book a 30-minute session with your manager. "
            "All sessions must be completed by January 31st."
        ),
        timestamp="2024-01-15T08:00:00Z",
        expected_priority="medium",
        expected_category="hr",
        expected_route="hr",
    ),
    EmailWithGroundTruth(
        id="med_003",
        sender="support@it-helpdesk.acmecorp.com",
        subject="Your password expires in 3 days",
        body=(
            "This is a reminder that your corporate password will expire in 3 days. "
            "Please log in to the IT portal to reset your password before it expires "
            "to avoid account lockout."
        ),
        timestamp="2024-01-15T08:15:00Z",
        expected_priority="medium",
        expected_category="it_support",
        expected_route="it",
    ),
    EmailWithGroundTruth(
        id="med_004",
        sender="securityalert@acme-corp-security.net",
        subject="⚠️ Suspicious Login Detected — Verify Your Account NOW",
        body=(
            "We detected a login attempt from an unknown device. "
            "Click here IMMEDIATELY to verify your identity and protect your account: "
            "http://acme-corp-verify.suspicious-domain.xyz/login "
            "Failure to verify within 1 hour will result in account suspension."
        ),
        timestamp="2024-01-15T08:30:00Z",
        expected_priority="urgent",
        expected_category="phishing",
        expected_route="security",
    ),
    EmailWithGroundTruth(
        id="med_005",
        sender="dev-team@acmecorp.com",
        subject="Sprint 42 Retrospective Notes",
        body=(
            "Hi all, here are the action items from Sprint 42 retro: "
            "1) Fix CI/CD pipeline flakiness (owner: Alex). "
            "2) Add integration tests for payment module (owner: Maria). "
            "3) Schedule architecture review for Q1 (owner: TBD). "
            "Please update JIRA tickets by EOD."
        ),
        timestamp="2024-01-15T09:00:00Z",
        expected_priority="medium",
        expected_category="internal_task",
        expected_route="manager",
    ),
    EmailWithGroundTruth(
        id="med_006",
        sender="promo@shopnow-weekly.com",
        subject="Your Weekly Deals Inside!",
        body=(
            "Check out this week's top deals hand-picked for you! "
            "Electronics, home goods, fashion and more — all on sale. "
            "Shop now and save big!"
        ),
        timestamp="2024-01-15T09:15:00Z",
        expected_priority="low",
        expected_category="spam",
        expected_route="trash",
    ),
]


# ─────────────────────────────────────────────
# HARD TASK — 12 emails, phishing, ambiguous, urgent
# ─────────────────────────────────────────────

HARD_EMAILS: List[EmailWithGroundTruth] = [
    EmailWithGroundTruth(
        id="hard_001",
        sender="ceo@acmecorp.com",
        subject="CRITICAL: Wire Transfer Required — Acquisition Closing Today",
        body=(
            "I'm in a board meeting and cannot take calls. We are closing the TechVentures "
            "acquisition TODAY. You must initiate a wire transfer of $2.4M to the escrow "
            "account immediately. Details: Account 847291047, Routing 021000021. "
            "Do NOT discuss this with anyone else. Reply confirmation only."
        ),
        timestamp="2024-01-15T06:00:00Z",
        expected_priority="urgent",
        expected_category="phishing",
        expected_route="security",
    ),
    EmailWithGroundTruth(
        id="hard_002",
        sender="legal@acmecorp.com",
        subject="NDA Signature Required — Partner Agreement Expires Tomorrow",
        body=(
            "Please review and sign the attached NDA for the DataStream partnership. "
            "The agreement expires tomorrow at midnight. Legal has reviewed and approved. "
            "Use DocuSign link: https://docusign.acmecorp.com/nda-datastream-2024"
        ),
        timestamp="2024-01-15T07:00:00Z",
        expected_priority="high",
        expected_category="legal",
        expected_route="executive",
    ),
    EmailWithGroundTruth(
        id="hard_003",
        sender="noreply@microsoft-security-alert.info",
        subject="Microsoft Account: Unusual Activity Detected",
        body=(
            "Dear Microsoft user, we detected unusual sign-in activity on your account. "
            "To secure your account, please verify your credentials immediately: "
            "https://microsofT-login.info/verify?token=abc123 "
            "This link expires in 30 minutes."
        ),
        timestamp="2024-01-15T07:15:00Z",
        expected_priority="urgent",
        expected_category="phishing",
        expected_route="security",
    ),
    EmailWithGroundTruth(
        id="hard_004",
        sender="cto@acmecorp.com",
        subject="Production Outage — All Hands Required",
        body=(
            "We have a P0 production outage affecting 40% of customers. Payment processing "
            "is down. Engineering, DevOps, and SRE leads: join the war room immediately. "
            "Bridge: https://meet.acmecorp.com/warroom-p0. This is our highest priority."
        ),
        timestamp="2024-01-15T07:30:00Z",
        expected_priority="urgent",
        expected_category="urgent_business",
        expected_route="executive",
    ),
    EmailWithGroundTruth(
        id="hard_005",
        sender="hr@acmecorp.com",
        subject="Benefits Open Enrollment Closes This Week",
        body=(
            "Reminder: Open enrollment for health, dental, and vision benefits closes "
            "this Friday. Log in to the HR portal to make your selections. "
            "Employees who do not enroll will remain on their current plan."
        ),
        timestamp="2024-01-15T08:00:00Z",
        expected_priority="medium",
        expected_category="hr",
        expected_route="hr",
    ),
    EmailWithGroundTruth(
        id="hard_006",
        sender="finance-alerts@acmecorp.com",
        subject="Invoice #INV-2024-0892 Overdue — Vendor Escalation",
        body=(
            "Vendor InfraCloud Solutions has escalated invoice #INV-2024-0892 ($47,500) "
            "which is now 45 days overdue. They have indicated they will suspend service "
            "within 48 hours if payment is not received. Requires immediate AP review."
        ),
        timestamp="2024-01-15T08:30:00Z",
        expected_priority="high",
        expected_category="finance",
        expected_route="finance",
    ),
    EmailWithGroundTruth(
        id="hard_007",
        sender="alice.johnson@acmecorp.com",
        subject="Can you help review my PR?",
        body=(
            "Hey, I've got a PR up for the new authentication module. "
            "It's been sitting for 3 days and I need at least one more approval to merge. "
            "Link: https://github.com/acme/repo/pull/1847. No rush, but would appreciate "
            "a look when you get a chance."
        ),
        timestamp="2024-01-15T09:00:00Z",
        expected_priority="low",
        expected_category="internal_task",
        expected_route="manager",
    ),
    EmailWithGroundTruth(
        id="hard_008",
        sender="awards@industry-recognition-committee.org",
        subject="Congratulations! Your Company Has Been Selected for an Award",
        body=(
            "Dear Business Leader, your company has been selected for the 2024 Industry "
            "Excellence Award! To claim your award and featured listing, please complete "
            "the registration form and pay the $299 listing fee at the link below."
        ),
        timestamp="2024-01-15T09:15:00Z",
        expected_priority="low",
        expected_category="spam",
        expected_route="trash",
    ),
    EmailWithGroundTruth(
        id="hard_009",
        sender="compliance@acmecorp.com",
        subject="SOC 2 Audit Evidence Collection — Response Required by Jan 20",
        body=(
            "As part of our SOC 2 Type II audit, we need evidence from your team by "
            "January 20th. Required: access logs for the past 90 days, change management "
            "records, and incident response documentation. Please coordinate with your "
            "team leads and submit via the compliance portal."
        ),
        timestamp="2024-01-15T09:30:00Z",
        expected_priority="high",
        expected_category="legal",
        expected_route="manager",
    ),
    EmailWithGroundTruth(
        id="hard_010",
        sender="ceo@acmecorp-communications.net",
        subject="Confidential: Restructuring Plans — Do Not Forward",
        body=(
            "As we prepare for Q1 restructuring, I need you to pull together a list of "
            "employees by department with their current compensation. Please send this "
            "to me at this email address only — not my corporate email. Keep this strictly "
            "confidential. The board has approved this request."
        ),
        timestamp="2024-01-15T10:00:00Z",
        expected_priority="urgent",
        expected_category="phishing",
        expected_route="security",
    ),
    EmailWithGroundTruth(
        id="hard_011",
        sender="it-security@acmecorp.com",
        subject="Mandatory Security Training — Complete by January 31",
        body=(
            "All employees must complete the annual security awareness training by "
            "January 31. Training takes approximately 45 minutes. Log in to the "
            "Learning Management System to begin: https://lms.acmecorp.com/security-2024"
        ),
        timestamp="2024-01-15T10:30:00Z",
        expected_priority="medium",
        expected_category="it_support",
        expected_route="it",
    ),
    EmailWithGroundTruth(
        id="hard_012",
        sender="marketing@acmecorp.com",
        subject="Q1 Campaign Launch — Feedback Needed",
        body=(
            "Hi team, the Q1 marketing campaign is ready for internal review before launch. "
            "Please review the campaign brief and creative assets in the shared drive and "
            "provide feedback by Thursday. Your input helps us ensure brand consistency."
        ),
        timestamp="2024-01-15T11:00:00Z",
        expected_priority="medium",
        expected_category="internal_task",
        expected_route="manager",
    ),
]


# ─────────────────────────────────────────────
# Task Registry
# ─────────────────────────────────────────────

TASK_EMAIL_MAP: Dict[str, List[EmailWithGroundTruth]] = {
    "easy": EASY_EMAILS,
    "medium": MEDIUM_EMAILS,
    "hard": HARD_EMAILS,
}

TASK_MAX_STEPS: Dict[str, int] = {
    "easy": 10,
    "medium": 20,
    "hard": 40,
}


def get_emails_for_task(task_id: str) -> List[EmailWithGroundTruth]:
    """Return the email list for a given task ID."""
    if task_id not in TASK_EMAIL_MAP:
        raise ValueError(f"Unknown task_id: {task_id}. Must be one of {list(TASK_EMAIL_MAP.keys())}")
    return TASK_EMAIL_MAP[task_id]


def get_email_by_id(task_id: str, email_id: str) -> EmailWithGroundTruth:
    """Return a specific email from a task by its ID."""
    emails = get_emails_for_task(task_id)
    for email in emails:
        if email.id == email_id:
            return email
    raise ValueError(f"Email '{email_id}' not found in task '{task_id}'")
