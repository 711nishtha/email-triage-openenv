"""
data.py — Synthetic realistic email generator for the Enterprise Email Triage OpenEnv.

Generates diverse, realistic email scenarios across three difficulty tiers.
All data is entirely fictional. No real PII is used.
"""

from __future__ import annotations

import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from models import EmailMessage, SenderProfile


# ── Synthetic sender registry ────────────────────────────────────────────────

INTERNAL_DOMAIN = "acmecorp.com"

SENDER_REGISTRY: Dict[str, SenderProfile] = {
    # C-Suite
    f"ceo@{INTERNAL_DOMAIN}": SenderProfile(
        email=f"ceo@{INTERNAL_DOMAIN}", display_name="Sarah Chen",
        domain=INTERNAL_DOMAIN, is_internal=True, is_vip=True,
        reputation_score=0.99, previous_interactions=120,
        department="Executive", job_title="CEO",
    ),
    f"cfo@{INTERNAL_DOMAIN}": SenderProfile(
        email=f"cfo@{INTERNAL_DOMAIN}", display_name="Marcus Webb",
        domain=INTERNAL_DOMAIN, is_internal=True, is_vip=True,
        reputation_score=0.99, previous_interactions=85,
        department="Finance", job_title="CFO",
    ),
    f"cto@{INTERNAL_DOMAIN}": SenderProfile(
        email=f"cto@{INTERNAL_DOMAIN}", display_name="Priya Nair",
        domain=INTERNAL_DOMAIN, is_internal=True, is_vip=True,
        reputation_score=0.99, previous_interactions=95,
        department="Technology", job_title="CTO",
    ),
    # HR
    f"hr-director@{INTERNAL_DOMAIN}": SenderProfile(
        email=f"hr-director@{INTERNAL_DOMAIN}", display_name="Linda Kowalski",
        domain=INTERNAL_DOMAIN, is_internal=True,
        reputation_score=0.97, previous_interactions=40,
        department="HR", job_title="HR Director",
    ),
    # Engineering
    f"eng-lead@{INTERNAL_DOMAIN}": SenderProfile(
        email=f"eng-lead@{INTERNAL_DOMAIN}", display_name="James Osei",
        domain=INTERNAL_DOMAIN, is_internal=True,
        reputation_score=0.96, previous_interactions=200,
        department="Engineering", job_title="Engineering Lead",
    ),
    # IT
    f"it-support@{INTERNAL_DOMAIN}": SenderProfile(
        email=f"it-support@{INTERNAL_DOMAIN}", display_name="IT Support Team",
        domain=INTERNAL_DOMAIN, is_internal=True,
        reputation_score=0.98, previous_interactions=300,
        department="IT", job_title="IT Support",
    ),
    # Known vendors
    "contracts@cloudprovider.com": SenderProfile(
        email="contracts@cloudprovider.com", display_name="CloudProvider Contracts",
        domain="cloudprovider.com", is_internal=False, is_known_vendor=True,
        reputation_score=0.92, previous_interactions=15,
        department="Vendor", job_title="Account Manager",
    ),
    "support@legit-saas.io": SenderProfile(
        email="support@legit-saas.io", display_name="LegitSaaS Support",
        domain="legit-saas.io", is_internal=False, is_known_vendor=True,
        reputation_score=0.88, previous_interactions=8,
    ),
    # Board member
    "r.patel@boardmembers.acmecorp.com": SenderProfile(
        email="r.patel@boardmembers.acmecorp.com", display_name="Raj Patel",
        domain="boardmembers.acmecorp.com", is_internal=False, is_vip=True,
        reputation_score=0.95, previous_interactions=20,
        job_title="Board Member",
    ),
    # Suspicious / phishing domains
    f"ceo@acmecorp-security.com": SenderProfile(
        email=f"ceo@acmecorp-security.com", display_name="Sarah Chen",  # display name spoofed
        domain="acmecorp-security.com", is_internal=False,
        reputation_score=0.05, previous_interactions=0,
        is_flagged_suspicious=True,
    ),
    "it-helpdesk@acme-corp.net": SenderProfile(
        email="it-helpdesk@acme-corp.net", display_name="IT Helpdesk",
        domain="acme-corp.net", is_internal=False,
        reputation_score=0.04, previous_interactions=0,
        is_flagged_suspicious=True,
    ),
    "noreply@newsletter-blast.com": SenderProfile(
        email="noreply@newsletter-blast.com", display_name="TechWeekly",
        domain="newsletter-blast.com", is_internal=False,
        reputation_score=0.70, previous_interactions=52,
    ),
}


def _make_id() -> str:
    return str(uuid.uuid4())[:8]


def _ts(days_ago: float = 0, hours_ago: float = 0, minutes_ago: float = 0) -> datetime:
    return datetime.utcnow() - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)


# ── Easy task emails ─────────────────────────────────────────────────────────

def make_easy_inbox(seed: Optional[int] = None) -> Tuple[List[EmailMessage], Dict[str, SenderProfile]]:
    """Single clear email — basic categorise + priority."""
    rng = random.Random(seed)
    choices = [_easy_it_request, _easy_team_update, _easy_vendor_followup]
    fn = rng.choice(choices)
    email = fn()
    senders = {email.sender: SENDER_REGISTRY.get(email.sender, _unknown_profile(email.sender))}
    return [email], senders


def _easy_it_request() -> EmailMessage:
    eid = _make_id()
    email = EmailMessage(
        email_id=eid,
        subject="Laptop replacement request — display flickering",
        sender=f"eng-lead@{INTERNAL_DOMAIN}",
        sender_display_name="James Osei",
        recipients=[f"it-support@{INTERNAL_DOMAIN}"],
        timestamp=_ts(hours_ago=2),
        body=(
            "Hi IT team,\n\n"
            "My laptop display has been flickering for the past three days. "
            "I've tried the usual fixes (reconnecting the display cable, updating drivers) "
            "but the issue persists. Could someone arrange a replacement or schedule a repair? "
            "I'm blocked on a few tasks that require an external monitor.\n\n"
            "Happy to drop it off at the IT desk whenever is convenient.\n\n"
            "Thanks,\nJames"
        ),
        has_attachments=False,
    )
    # Attach hidden ground-truth
    object.__setattr__(email, '_true_priority', 'medium')
    object.__setattr__(email, '_true_category', 'it_support')
    object.__setattr__(email, '_true_route', 'it_helpdesk')
    object.__setattr__(email, '_is_phishing', False)
    return email


def _easy_team_update() -> EmailMessage:
    eid = _make_id()
    email = EmailMessage(
        email_id=eid,
        subject="Sprint 47 retrospective notes — action items inside",
        sender=f"eng-lead@{INTERNAL_DOMAIN}",
        sender_display_name="James Osei",
        recipients=[f"engineering@{INTERNAL_DOMAIN}"],
        timestamp=_ts(hours_ago=1),
        body=(
            "Team,\n\n"
            "Please find below the action items from our Sprint 47 retro:\n"
            "1. Fix flaky integration tests in the payments module by EOW (owner: Dev).\n"
            "2. Update runbooks for the new auth service deployment (owner: Ops).\n"
            "3. Schedule a knowledge-transfer session on the new caching layer (owner: James).\n\n"
            "Full notes in Confluence: https://wiki.acmecorp.com/sprint47-retro\n\n"
            "Best,\nJames"
        ),
        links=["https://wiki.acmecorp.com/sprint47-retro"],
    )
    object.__setattr__(email, '_true_priority', 'low')
    object.__setattr__(email, '_true_category', 'team_update')
    object.__setattr__(email, '_true_route', 'engineering_team')
    object.__setattr__(email, '_is_phishing', False)
    return email


def _easy_vendor_followup() -> EmailMessage:
    eid = _make_id()
    email = EmailMessage(
        email_id=eid,
        subject="Re: Cloud infrastructure contract renewal — Q1 pricing",
        sender="contracts@cloudprovider.com",
        sender_display_name="CloudProvider Contracts",
        recipients=[f"cfo@{INTERNAL_DOMAIN}"],
        timestamp=_ts(hours_ago=4),
        body=(
            "Dear Marcus,\n\n"
            "Following up on our conversation from last week. "
            "I've attached the updated Q1 pricing sheet for the infrastructure contract renewal. "
            "The revised committed-use discount brings the annual total to $420,000, "
            "down from the current $480,000.\n\n"
            "Could your team review by end of month so we can lock in the rate before "
            "our pricing model refreshes on the 1st?\n\n"
            "Best regards,\nAlex Turner\nAccount Manager, CloudProvider"
        ),
        has_attachments=True,
        attachment_names=["Q1_pricing_sheet_v2.pdf"],
    )
    object.__setattr__(email, '_true_priority', 'high')
    object.__setattr__(email, '_true_category', 'vendor_contract')
    object.__setattr__(email, '_true_route', 'finance_team')
    object.__setattr__(email, '_is_phishing', False)
    return email


# ── Medium task emails ───────────────────────────────────────────────────────

def make_medium_inbox(seed: Optional[int] = None) -> Tuple[
    List[EmailMessage],
    Dict[str, List[EmailMessage]],
    Dict[str, SenderProfile],
]:
    """Five emails with thread context, calendar check, KB lookup."""
    _ = random.Random(seed)

    thread_id_a = f"thread-{_make_id()}"
    thread_id_b = f"thread-{_make_id()}"

    # Prior thread history (context for agent)
    prior_thread_a = _medium_thread_history_a(thread_id_a)
    prior_thread_b = _medium_thread_history_b(thread_id_b)

    inbox = [
        _medium_calendar_followup(thread_id_a),
        _medium_vendor_kb_request(),
        _medium_hr_matter(),
        _medium_ceo_brief_request(),
        _medium_newsletter_lookalike(),
    ]

    thread_history = {
        thread_id_a: prior_thread_a,
        thread_id_b: prior_thread_b,
    }
    # Link second thread to calendar email
    inbox[0] = inbox[0].model_copy(update={"thread_id": thread_id_a})

    senders: Dict[str, SenderProfile] = {}
    for email in inbox:
        senders[email.sender] = SENDER_REGISTRY.get(
            email.sender, _unknown_profile(email.sender)
        )

    return inbox, thread_history, senders


def _medium_thread_history_a(thread_id: str) -> List[EmailMessage]:
    return [
        EmailMessage(
            email_id=_make_id(), thread_id=thread_id,
            subject="Board presentation — scheduling",
            sender=f"ceo@{INTERNAL_DOMAIN}", sender_display_name="Sarah Chen",
            recipients=[f"executive-assistant@{INTERNAL_DOMAIN}"],
            timestamp=_ts(days_ago=3),
            body="Can you book the Boardroom A for Thursday 2 PM? We need 2 hours.",
        ),
        EmailMessage(
            email_id=_make_id(), thread_id=thread_id,
            subject="Re: Board presentation — scheduling",
            sender=f"executive-assistant@{INTERNAL_DOMAIN}",
            sender_display_name="EA Team",
            recipients=[f"ceo@{INTERNAL_DOMAIN}"],
            timestamp=_ts(days_ago=2),
            body="Boardroom A is available Thursday 2–4 PM. Confirmed and calendar invite sent.",
            is_reply=True,
        ),
    ]


def _medium_thread_history_b(thread_id: str) -> List[EmailMessage]:
    return [
        EmailMessage(
            email_id=_make_id(), thread_id=thread_id,
            subject="SaaS onboarding — data migration questions",
            sender="support@legit-saas.io", sender_display_name="LegitSaaS Support",
            recipients=[f"eng-lead@{INTERNAL_DOMAIN}"],
            timestamp=_ts(days_ago=5),
            body="Hi, we need your database schema documentation before we can start the migration.",
        ),
    ]


def _medium_calendar_followup(thread_id: str) -> EmailMessage:
    eid = _make_id()
    email = EmailMessage(
        email_id=eid, thread_id=thread_id,
        subject="Re: Board presentation — need room change",
        sender=f"ceo@{INTERNAL_DOMAIN}", sender_display_name="Sarah Chen",
        recipients=[f"executive-assistant@{INTERNAL_DOMAIN}"],
        timestamp=_ts(hours_ago=1),
        body=(
            "The board has grown — we now have 12 attendees. "
            "Boardroom A only fits 10. Can you check if the Executive Suite is free Thursday 2–4 PM? "
            "If not, we'll need to find an external venue — check the approved vendor list in the KB."
        ),
        is_reply=True,
    )
    object.__setattr__(email, '_true_priority', 'high')
    object.__setattr__(email, '_true_category', 'executive_request')
    object.__setattr__(email, '_true_route', 'executive_assistant')
    object.__setattr__(email, '_is_phishing', False)
    return email


def _medium_vendor_kb_request() -> EmailMessage:
    eid = _make_id()
    email = EmailMessage(
        email_id=eid,
        subject="Contract amendment — GDPR data processing addendum required",
        sender="contracts@cloudprovider.com", sender_display_name="CloudProvider Contracts",
        recipients=[f"legal@{INTERNAL_DOMAIN}"],
        timestamp=_ts(hours_ago=3),
        body=(
            "Hi team,\n\nFollowing recent EU regulatory updates, our legal team requires "
            "a signed Data Processing Addendum (DPA) before we can process any data "
            "in EU regions under the new contract. "
            "Could you confirm whether your standard DPA template covers Article 28 SCCs? "
            "If not, we have a template we can share.\n\nBest,\nAlex"
        ),
    )
    object.__setattr__(email, '_true_priority', 'high')
    object.__setattr__(email, '_true_category', 'vendor_contract')
    object.__setattr__(email, '_true_route', 'legal_team')
    object.__setattr__(email, '_is_phishing', False)
    return email


def _medium_hr_matter() -> EmailMessage:
    eid = _make_id()
    email = EmailMessage(
        email_id=eid,
        subject="Confidential: Grievance submission — Engineering team",
        sender=f"hr-director@{INTERNAL_DOMAIN}", sender_display_name="Linda Kowalski",
        recipients=[f"ceo@{INTERNAL_DOMAIN}"],
        timestamp=_ts(hours_ago=2),
        body=(
            "Dear Sarah,\n\n"
            "I'm forwarding a formal grievance submitted by a member of the Engineering team "
            "regarding workplace conduct. As per policy, this requires your acknowledgment "
            "within 48 hours. Please treat this as confidential.\n\n"
            "Full details are in the attached encrypted document. Password sent separately.\n\n"
            "Linda"
        ),
        has_attachments=True,
        attachment_names=["grievance_ref_2024_0047_encrypted.pdf"],
    )
    object.__setattr__(email, '_true_priority', 'critical')
    object.__setattr__(email, '_true_category', 'hr_matter')
    object.__setattr__(email, '_true_route', 'ceo_office')
    object.__setattr__(email, '_is_phishing', False)
    return email


def _medium_ceo_brief_request() -> EmailMessage:
    eid = _make_id()
    email = EmailMessage(
        email_id=eid,
        subject="Request for Q3 board brief — needed by Thursday",
        sender=f"ceo@{INTERNAL_DOMAIN}", sender_display_name="Sarah Chen",
        recipients=[f"cfo@{INTERNAL_DOMAIN}", f"cto@{INTERNAL_DOMAIN}"],
        timestamp=_ts(hours_ago=5),
        body=(
            "Marcus, Priya —\n\n"
            "I need a combined financial and technical brief for the board meeting Thursday. "
            "Please coordinate and have a draft in my inbox by Wednesday EOD.\n\n"
            "Sarah"
        ),
    )
    object.__setattr__(email, '_true_priority', 'critical')
    object.__setattr__(email, '_true_category', 'executive_request')
    object.__setattr__(email, '_true_route', 'cfo_office')
    object.__setattr__(email, '_is_phishing', False)
    return email


def _medium_newsletter_lookalike() -> EmailMessage:
    eid = _make_id()
    email = EmailMessage(
        email_id=eid,
        subject="🚨 THIS WEEK: Your mandatory security training expires TOMORROW",
        sender="noreply@newsletter-blast.com",
        sender_display_name="TechWeekly",
        recipients=[f"all-staff@{INTERNAL_DOMAIN}"],
        timestamp=_ts(hours_ago=6),
        body=(
            "URGENT NOTICE\n\n"
            "Your annual cybersecurity awareness training certification expires TOMORROW. "
            "Click below to complete your training and avoid access suspension:\n"
            "https://newsletter-blast.com/training-redirect?uid=12345\n\n"
            "— The TechWeekly Team"
        ),
        links=["https://newsletter-blast.com/training-redirect?uid=12345"],
    )
    object.__setattr__(email, '_true_priority', 'low')
    object.__setattr__(email, '_true_category', 'newsletter')
    object.__setattr__(email, '_true_route', 'archive')
    object.__setattr__(email, '_is_phishing', False)  # suspicious but technically newsletter
    return email


# ── Hard task emails ─────────────────────────────────────────────────────────

def make_hard_inbox(seed: Optional[int] = None) -> Tuple[
    List[EmailMessage],
    Dict[str, List[EmailMessage]],
    Dict[str, SenderProfile],
]:
    """Eight emails: phishing, BEC, real security incident, legitimate urgent, noise."""

    inbox = [
        _hard_phishing_ceo_spoof(),
        _hard_phishing_it_password_reset(),
        _hard_real_security_incident(),
        _hard_bec_wire_transfer(),
        _hard_legitimate_urgent_a(),
        _hard_legitimate_urgent_b(),
        _hard_legitimate_urgent_c(),
        _hard_urgent_newsletter(),
    ]

    # Shuffle to prevent positional bias in grading
    rng = random.Random(seed)
    rng.shuffle(inbox)

    senders: Dict[str, SenderProfile] = {}
    for email in inbox:
        senders[email.sender] = SENDER_REGISTRY.get(
            email.sender, _unknown_profile(email.sender)
        )

    return inbox, {}, senders


def _hard_phishing_ceo_spoof() -> EmailMessage:
    eid = _make_id()
    email = EmailMessage(
        email_id=eid,
        subject="URGENT: Wire transfer authorisation needed immediately",
        sender=f"ceo@acmecorp-security.com",   # spoofed domain!
        sender_display_name="Sarah Chen",       # display name matches real CEO
        recipients=[f"finance@{INTERNAL_DOMAIN}"],
        timestamp=_ts(hours_ago=0.5),
        body=(
            "Hi,\n\n"
            "I need you to process an urgent wire transfer of $87,500 to a new supplier "
            "we're onboarding for the Q4 project. This is time-sensitive and must be done "
            "before market close today. Please treat this as confidential — do not discuss "
            "with others until the deal is public.\n\n"
            "Transfer to:\nBank: First National\nRouting: 021000021\nAccount: 8847291045\n\n"
            "Authorisation code: CEO-2024-Q4-0091\n\n"
            "Sarah"
        ),
    )
    object.__setattr__(email, '_true_priority', 'spam')
    object.__setattr__(email, '_true_category', 'phishing')
    object.__setattr__(email, '_true_route', 'security_team')
    object.__setattr__(email, '_is_phishing', True)
    object.__setattr__(email, '_is_bec', True)
    return email


def _hard_phishing_it_password_reset() -> EmailMessage:
    eid = _make_id()
    email = EmailMessage(
        email_id=eid,
        subject="Action Required: Your corporate password expires in 24 hours",
        sender="it-helpdesk@acme-corp.net",    # typosquatted domain
        sender_display_name="IT Helpdesk",
        recipients=[f"all-staff@{INTERNAL_DOMAIN}"],
        timestamp=_ts(hours_ago=2),
        body=(
            "Dear Employee,\n\n"
            "Your corporate account password is set to expire in 24 hours. "
            "To avoid being locked out of company systems, please reset your password immediately:\n\n"
            "https://acme-corp.net/password-reset?token=CORP2024RESET\n\n"
            "Enter your current username and password to authenticate, "
            "then you will be prompted to create a new password.\n\n"
            "If you do not reset within 24 hours, your account will be suspended "
            "and you will need to contact HR to reinstate access.\n\n"
            "IT Helpdesk Team"
        ),
        links=["https://acme-corp.net/password-reset?token=CORP2024RESET"],
    )
    object.__setattr__(email, '_true_priority', 'spam')
    object.__setattr__(email, '_true_category', 'phishing')
    object.__setattr__(email, '_true_route', 'security_team')
    object.__setattr__(email, '_is_phishing', True)
    object.__setattr__(email, '_is_bec', False)
    return email


def _hard_real_security_incident() -> EmailMessage:
    eid = _make_id()
    email = EmailMessage(
        email_id=eid,
        subject="[ALERT] Anomalous login detected — production DB cluster",
        sender=f"it-support@{INTERNAL_DOMAIN}",   # legitimate internal IT
        sender_display_name="IT Support Team",
        recipients=[f"cto@{INTERNAL_DOMAIN}", f"eng-lead@{INTERNAL_DOMAIN}"],
        timestamp=_ts(minutes_ago=15),
        body=(
            "SECURITY ALERT — Immediate Action Required\n\n"
            "Our SIEM has detected anomalous login activity on the production database cluster "
            "(prod-db-01, prod-db-02). Activity profile:\n\n"
            "- Source IP: 185.220.101.47 (flagged as Tor exit node)\n"
            "- Login attempts: 847 in last 10 minutes (brute-force pattern)\n"
            "- 3 successful authentications using service account 'db_readonly'\n"
            "- Data exfiltration pattern detected: 2.3 GB outbound transfer in progress\n\n"
            "Current status: Automated firewall rules have blocked the IP. "
            "Service account has been disabled. Investigation in progress.\n\n"
            "Immediate actions required:\n"
            "1. Confirm scope of accessed data (PII risk assessment)\n"
            "2. Rotate all service account credentials\n"
            "3. Notify CISO and legal team (potential breach notification obligation)\n\n"
            "— IT Security Monitoring"
        ),
    )
    object.__setattr__(email, '_true_priority', 'critical')
    object.__setattr__(email, '_true_category', 'security_incident')
    object.__setattr__(email, '_true_route', 'security_team')
    object.__setattr__(email, '_is_phishing', False)
    object.__setattr__(email, '_is_bec', False)
    return email


def _hard_bec_wire_transfer() -> EmailMessage:
    """BEC from a legitimate-seeming board member email requesting unusual wire."""
    eid = _make_id()
    email = EmailMessage(
        email_id=eid,
        subject="Confidential acquisition — urgent transfer",
        sender="r.patel@boardmembers.acmecorp.com",  # Legitimate domain, but BEC request
        sender_display_name="Raj Patel",
        recipients=[f"cfo@{INTERNAL_DOMAIN}"],
        timestamp=_ts(hours_ago=1),
        body=(
            "Marcus,\n\n"
            "I'm reaching out directly regarding a sensitive M&A opportunity "
            "we've been evaluating at the board level. I can't share details over email "
            "until the NDA is signed, but we need to move a $250,000 holding deposit "
            "to escrow by end of day to secure our position.\n\n"
            "Please do NOT discuss this with Sarah or the rest of the exec team yet — "
            "timing is critical and premature disclosure could kill the deal.\n\n"
            "Use the wire details I'll send in a separate follow-up.\n\n"
            "Raj"
        ),
    )
    object.__setattr__(email, '_true_priority', 'spam')
    object.__setattr__(email, '_true_category', 'phishing')
    object.__setattr__(email, '_true_route', 'security_team')
    object.__setattr__(email, '_is_phishing', True)
    object.__setattr__(email, '_is_bec', True)
    return email


def _hard_legitimate_urgent_a() -> EmailMessage:
    eid = _make_id()
    email = EmailMessage(
        email_id=eid,
        subject="Production outage — payments API returning 500s",
        sender=f"eng-lead@{INTERNAL_DOMAIN}", sender_display_name="James Osei",
        recipients=[f"cto@{INTERNAL_DOMAIN}", f"it-support@{INTERNAL_DOMAIN}"],
        timestamp=_ts(hours_ago=0.1),
        body=(
            "Priya, IT —\n\n"
            "Payments API has been returning 500 errors for the last 8 minutes. "
            "Error rate: 94%. Monitoring dashboard: https://monitoring.acmecorp.com/payments\n\n"
            "We've rolled back the last deployment but the errors persist. "
            "Potentially a database connection pool issue. "
            "On-call engineer is investigating but I'm looping you in given customer impact.\n\n"
            "James"
        ),
        links=["https://monitoring.acmecorp.com/payments"],
    )
    object.__setattr__(email, '_true_priority', 'critical')
    object.__setattr__(email, '_true_category', 'it_support')
    object.__setattr__(email, '_true_route', 'engineering_team')
    object.__setattr__(email, '_is_phishing', False)
    return email


def _hard_legitimate_urgent_b() -> EmailMessage:
    eid = _make_id()
    email = EmailMessage(
        email_id=eid,
        subject="URGENT: Key customer threatening churn — need exec call today",
        sender=f"sales@{INTERNAL_DOMAIN}", sender_display_name="Sales Team",
        recipients=[f"ceo@{INTERNAL_DOMAIN}"],
        timestamp=_ts(hours_ago=2),
        body=(
            "Sarah,\n\n"
            "GlobalRetail Inc (our #2 account, $1.8M ARR) has contacted us this morning "
            "threatening to terminate their contract. Their VP Engineering is unhappy "
            "with our SLA performance over Q3 (avg uptime 99.1% vs contracted 99.9%).\n\n"
            "They're requesting an exec-level call today. I've blocked 3–4 PM on your calendar "
            "as a placeholder. Can you confirm?\n\n"
            "— Sales"
        ),
    )
    object.__setattr__(email, '_true_priority', 'critical')
    object.__setattr__(email, '_true_category', 'customer_escalation')
    object.__setattr__(email, '_true_route', 'ceo_office')
    object.__setattr__(email, '_is_phishing', False)
    return email


def _hard_legitimate_urgent_c() -> EmailMessage:
    eid = _make_id()
    email = EmailMessage(
        email_id=eid,
        subject="Board resolution needed — regulatory filing deadline this Friday",
        sender="r.patel@boardmembers.acmecorp.com",
        sender_display_name="Raj Patel",
        recipients=[f"ceo@{INTERNAL_DOMAIN}", f"cfo@{INTERNAL_DOMAIN}"],
        timestamp=_ts(hours_ago=3),
        body=(
            "Sarah, Marcus,\n\n"
            "As a reminder, our annual regulatory filing with the SEC has a hard deadline "
            "this Friday at 5 PM EST. We need a signed board resolution by Thursday noon "
            "to accompany the filing. Legal has the draft — please review and countersign.\n\n"
            "This is time-sensitive; missing the deadline carries a $50K fine per day.\n\n"
            "Raj"
        ),
    )
    object.__setattr__(email, '_true_priority', 'critical')
    object.__setattr__(email, '_true_category', 'executive_request')
    object.__setattr__(email, '_true_route', 'legal_team')
    object.__setattr__(email, '_is_phishing', False)
    return email


def _hard_urgent_newsletter() -> EmailMessage:
    eid = _make_id()
    email = EmailMessage(
        email_id=eid,
        subject="🔴 BREAKING: Critical vulnerability in OpenSSL — patch NOW",
        sender="noreply@newsletter-blast.com",
        sender_display_name="TechWeekly Security Digest",
        recipients=[f"all-staff@{INTERNAL_DOMAIN}"],
        timestamp=_ts(hours_ago=4),
        body=(
            "BREAKING SECURITY NEWS\n\n"
            "A critical RCE vulnerability (CVSS 9.8) has been discovered in OpenSSL 3.x. "
            "Patch your systems immediately.\n\n"
            "Full details and patch links: https://newsletter-blast.com/openssl-cve-2024\n\n"
            "Subscribe to TechWeekly Premium for real-time alerts: "
            "https://newsletter-blast.com/premium\n\n"
            "— TechWeekly Security Team"
        ),
        links=[
            "https://newsletter-blast.com/openssl-cve-2024",
            "https://newsletter-blast.com/premium",
        ],
    )
    object.__setattr__(email, '_true_priority', 'low')  # It's a newsletter, not an internal alert
    object.__setattr__(email, '_true_category', 'newsletter')
    object.__setattr__(email, '_true_route', 'archive')
    object.__setattr__(email, '_is_phishing', False)
    return email


# ── Helpers ──────────────────────────────────────────────────────────────────

def _unknown_profile(email_addr: str) -> SenderProfile:
    domain = email_addr.split("@")[-1] if "@" in email_addr else "unknown"
    return SenderProfile(
        email=email_addr,
        display_name=email_addr,
        domain=domain,
        is_internal=domain == INTERNAL_DOMAIN,
        reputation_score=0.5,
    )
