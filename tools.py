"""
tools.py — Mock safe tools available to the triage agent.

All tools are deterministic, sandboxed, and return realistic synthetic data.
No external API calls. No secrets required.

Available tools:
  - calendar_check   : Check room/person availability
  - kb_search        : Search internal knowledge base
  - sender_lookup    : Look up sender reputation and history
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional


# ── Tool dispatcher ──────────────────────────────────────────────────────────

AVAILABLE_TOOLS: List[str] = ["calendar_check", "kb_search", "sender_lookup"]


def run_tool(tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch a tool call by name.

    Returns a dict with keys:
      - success (bool)
      - result  (tool-specific data)
      - error   (str if success is False)
    """
    dispatch = {
        "calendar_check": _tool_calendar_check,
        "kb_search": _tool_kb_search,
        "sender_lookup": _tool_sender_lookup,
    }

    if tool_name not in dispatch:
        return {
            "success": False,
            "result": None,
            "error": f"Unknown tool '{tool_name}'. Available: {AVAILABLE_TOOLS}",
        }

    try:
        result = dispatch[tool_name](params)
        return {"success": True, "result": result, "error": None}
    except Exception as exc:
        return {"success": False, "result": None, "error": str(exc)}


# ── Calendar check ───────────────────────────────────────────────────────────

# Synthetic room availability database
_ROOM_SCHEDULE: Dict[str, Dict[str, bool]] = {
    "boardroom_a": {
        "monday_9am": True, "monday_2pm": False,
        "tuesday_9am": True, "tuesday_2pm": True,
        "wednesday_9am": False, "wednesday_2pm": True,
        "thursday_9am": True, "thursday_2pm": False,   # Boardroom A busy Thursday 2PM
        "friday_9am": True, "friday_2pm": True,
    },
    "executive_suite": {
        "monday_9am": False, "monday_2pm": True,
        "tuesday_9am": True, "tuesday_2pm": False,
        "wednesday_9am": True, "wednesday_2pm": True,
        "thursday_9am": False, "thursday_2pm": True,   # Executive Suite free Thursday 2PM
        "friday_9am": True, "friday_2pm": False,
    },
    "conference_room_1": {
        "thursday_2pm": True,
        "friday_2pm": True,
    },
}

# Person availability (for scheduling meetings)
_PERSON_SCHEDULE: Dict[str, Dict[str, str]] = {
    f"ceo@acmecorp.com": {
        "thursday_2pm": "available",
        "thursday_3pm": "available",
        "friday_9am": "busy (board call)",
    },
    f"cfo@acmecorp.com": {
        "thursday_2pm": "available",
        "thursday_3pm": "available",
        "wednesday_eod": "deadline: Q3 board brief due",
    },
    f"cto@acmecorp.com": {
        "thursday_2pm": "busy (engineering all-hands)",
        "thursday_3pm": "available",
    },
}


def _tool_calendar_check(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check room or person availability.

    Params:
      resource_type: "room" | "person"
      resource_name: str (room name or email address)
      day: str (e.g., "thursday")
      time: str (e.g., "2pm")

    Returns availability info and alternatives if unavailable.
    """
    resource_type = params.get("resource_type", "room")
    resource_name = str(params.get("resource_name", "")).lower().replace(" ", "_")
    day = str(params.get("day", "")).lower()
    time = str(params.get("time", "")).lower().replace(":", "").replace(" ", "")

    slot_key = f"{day}_{time}"

    if resource_type == "room":
        schedule = _ROOM_SCHEDULE.get(resource_name, {})
        available = schedule.get(slot_key, True)  # Default available if not in our DB

        # Build alternatives
        alternatives = []
        if not available:
            for room, sched in _ROOM_SCHEDULE.items():
                if room != resource_name and sched.get(slot_key, True):
                    alternatives.append(room.replace("_", " ").title())

        return {
            "resource": resource_name.replace("_", " ").title(),
            "slot": f"{day.title()} {time.upper()}",
            "available": available,
            "alternatives": alternatives,
            "note": (
                "Room available — calendar invite can be sent."
                if available
                else f"Room is booked. Suggested alternatives: {', '.join(alternatives) or 'none found'}"
            ),
        }

    elif resource_type == "person":
        schedule = _PERSON_SCHEDULE.get(params.get("resource_name", ""), {})
        status = schedule.get(slot_key, "available")
        return {
            "person": params.get("resource_name"),
            "slot": f"{day.title()} {time.upper()}",
            "status": status,
        }

    return {"error": "resource_type must be 'room' or 'person'"}


# ── KB search ────────────────────────────────────────────────────────────────

_KB_ARTICLES: List[Dict[str, str]] = [
    {
        "id": "KB-001",
        "title": "Data Processing Addendum (DPA) — Standard Template",
        "category": "legal",
        "content": (
            "Acme Corp's standard DPA template covers GDPR Article 28 requirements and includes "
            "Standard Contractual Clauses (SCCs) for EU data transfers. "
            "The template is maintained by the Legal team and reviewed quarterly. "
            "Current version: DPA-v3.2 (updated Jan 2024). "
            "Contact legal@acmecorp.com to initiate a DPA signing workflow."
        ),
        "tags": ["gdpr", "dpa", "legal", "vendor", "contract", "scc"],
    },
    {
        "id": "KB-002",
        "title": "Approved External Venue Vendors — Executive Meetings",
        "category": "operations",
        "content": (
            "Approved venues for off-site executive meetings (pre-vetted, NDA on file): "
            "1. The Pinnacle Conference Centre — capacity 20, AV included. "
            "2. Horizon Business Hub — capacity 30, catering available. "
            "3. CityView Boardroom — capacity 15, preferred for board meetings. "
            "Contact operations@acmecorp.com to book. Lead time: 48 hours minimum."
        ),
        "tags": ["venue", "executive", "offsite", "boardroom", "meeting"],
    },
    {
        "id": "KB-003",
        "title": "Security Incident Response Procedure",
        "category": "security",
        "content": (
            "Severity 1 (Critical): Data breach, active intrusion, ransomware. "
            "Immediate steps: (1) Notify security@acmecorp.com + CISO. "
            "(2) Isolate affected systems. (3) Preserve logs. "
            "(4) Notify legal team if PII is involved (breach notification SLA: 72 hours). "
            "Do NOT discuss publicly until authorised by Communications team."
        ),
        "tags": ["security", "incident", "breach", "response", "procedure"],
    },
    {
        "id": "KB-004",
        "title": "Phishing Reporting Procedure",
        "category": "security",
        "content": (
            "If you receive a suspected phishing email: "
            "(1) Do NOT click any links or download attachments. "
            "(2) Forward the email as attachment to phishing@acmecorp.com. "
            "(3) IT Security will confirm within 2 hours. "
            "Signs of phishing: urgency, spoofed domains, credential requests, wire transfers."
        ),
        "tags": ["phishing", "security", "email", "spoofing"],
    },
    {
        "id": "KB-005",
        "title": "Wire Transfer Authorisation Policy",
        "category": "finance",
        "content": (
            "All wire transfers over $10,000 require dual authorisation from CFO and CEO. "
            "New vendors must be verified through the Finance team's vendor onboarding process "
            "(minimum 3 business days). "
            "Email-only wire requests are NEVER valid — always confirm via phone with the requestor. "
            "Report suspicious wire requests to security@acmecorp.com immediately."
        ),
        "tags": ["wire", "transfer", "finance", "policy", "fraud", "bec"],
    },
    {
        "id": "KB-006",
        "title": "Employee Password Reset Procedure",
        "category": "it",
        "content": (
            "Password resets are initiated ONLY through the internal IT portal: "
            "https://it.acmecorp.com/reset (internal network only). "
            "IT will NEVER ask for your current password via email. "
            "If you receive a password reset email from an external domain, report it as phishing."
        ),
        "tags": ["password", "reset", "it", "security", "credential"],
    },
]


def _tool_kb_search(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Search the internal knowledge base.

    Params:
      query: str — search terms
      category: str (optional) — filter by category (legal, security, finance, it, operations)
      max_results: int (default 3)

    Returns list of matching KB articles (id, title, content snippet).
    """
    query = str(params.get("query", "")).lower()
    category_filter = str(params.get("category", "")).lower()
    max_results = int(params.get("max_results", 3))

    if not query:
        return {"results": [], "message": "No query provided."}

    query_terms = set(query.split())
    results = []

    for article in _KB_ARTICLES:
        # Apply category filter if specified
        if category_filter and article["category"] != category_filter:
            continue

        # Score relevance: keyword overlap with tags + title + content
        score = 0
        searchable = (
            " ".join(article["tags"])
            + " " + article["title"].lower()
            + " " + article["content"].lower()
        )
        for term in query_terms:
            if term in searchable:
                score += 1

        if score > 0:
            results.append({
                "id": article["id"],
                "title": article["title"],
                "category": article["category"],
                "snippet": article["content"][:250] + "...",
                "relevance_score": score,
            })

    # Sort by relevance
    results.sort(key=lambda x: x["relevance_score"], reverse=True)
    results = results[:max_results]

    return {
        "query": query,
        "results": results,
        "total_found": len(results),
        "message": f"Found {len(results)} matching article(s).",
    }


# ── Sender lookup ────────────────────────────────────────────────────────────

# Suspicious domain patterns (deterministic heuristics)
_SUSPICIOUS_PATTERNS: List[str] = [
    "acme-corp",        # typosquatting
    "acmecorp-",        # subdomain spoofing
    "acmec0rp",         # character substitution
    "acme_corp",        # underscore domain (illegal but some mail systems accept)
]

_KNOWN_TRUSTED_DOMAINS: List[str] = [
    "acmecorp.com",
    "boardmembers.acmecorp.com",
    "cloudprovider.com",
    "legit-saas.io",
]


def _tool_sender_lookup(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Look up sender reputation and flag suspicious domains.

    Params:
      email: str — sender email address

    Returns reputation data, domain analysis, and threat signals.
    """
    from models import SENDER_REGISTRY  # lazy import to avoid circular

    email = str(params.get("email", "")).lower().strip()
    if not email or "@" not in email:
        return {"error": "Valid email address required."}

    domain = email.split("@")[-1]

    # Check registry
    if email in SENDER_REGISTRY:
        profile = SENDER_REGISTRY[email]
        return {
            "email": email,
            "domain": domain,
            "known_sender": True,
            "display_name": profile.display_name,
            "is_internal": profile.is_internal,
            "is_vip": profile.is_vip,
            "reputation_score": profile.reputation_score,
            "previous_interactions": profile.previous_interactions,
            "job_title": profile.job_title,
            "department": profile.department,
            "is_flagged_suspicious": profile.is_flagged_suspicious,
            "threat_signals": (
                ["Domain pre-flagged as suspicious by security tooling"]
                if profile.is_flagged_suspicious else []
            ),
            "recommendation": (
                "DO NOT ENGAGE — report to security team"
                if profile.is_flagged_suspicious
                else "Trusted sender"
            ),
        }

    # Unknown sender — run heuristic analysis
    threat_signals = []

    # Check for typosquatting patterns
    for pattern in _SUSPICIOUS_PATTERNS:
        if pattern in domain:
            threat_signals.append(f"Domain contains suspicious pattern: '{pattern}'")

    # Check if domain spoofs a trusted domain name
    for trusted in _KNOWN_TRUSTED_DOMAINS:
        trusted_base = trusted.split(".")[0]
        if trusted_base in domain and domain != trusted:
            threat_signals.append(
                f"Domain '{domain}' may be spoofing trusted domain '{trusted}'"
            )

    # Heuristic reputation (unknown domains default to moderate suspicion)
    rep_score = 0.3 if threat_signals else 0.5

    return {
        "email": email,
        "domain": domain,
        "known_sender": False,
        "display_name": None,
        "is_internal": domain == "acmecorp.com",
        "is_vip": False,
        "reputation_score": rep_score,
        "previous_interactions": 0,
        "threat_signals": threat_signals,
        "recommendation": (
            "HIGH RISK — likely phishing or spoofed domain. Report to security."
            if threat_signals
            else "Unknown sender — proceed with caution."
        ),
    }


# Expose the registry reference for sender_lookup
try:
    from data import SENDER_REGISTRY
except ImportError:
    SENDER_REGISTRY = {}  # type: ignore
