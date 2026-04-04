"""
tools.py — Agent tools for email investigation and lookup.
All tools are deterministic and return structured results.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────
# Tool Definitions (for observation)
# ─────────────────────────────────────────────

AVAILABLE_TOOLS: List[str] = [
    "lookup_sender",
    "analyze_links",
    "check_sender_domain",
    "search_email_history",
    "flag_security_incident",
]


TOOL_DESCRIPTIONS: Dict[str, str] = {
    "lookup_sender": "Look up information about an email sender in the corporate directory.",
    "analyze_links": "Analyze URLs in an email for suspicious patterns.",
    "check_sender_domain": "Verify whether the sender's domain matches known corporate domains.",
    "search_email_history": "Search historical emails from this sender.",
    "flag_security_incident": "Flag an email as a security incident and notify the security team.",
}


# ─────────────────────────────────────────────
# Mock Corporate Directory
# ─────────────────────────────────────────────

CORPORATE_DIRECTORY: Dict[str, Dict[str, Any]] = {
    "ceo@acmecorp.com": {"name": "James Wilson", "role": "CEO", "department": "Executive", "verified": True},
    "cfo@acmecorp.com": {"name": "Sarah Chen", "role": "CFO", "department": "Finance", "verified": True},
    "cto@acmecorp.com": {"name": "Michael Torres", "role": "CTO", "department": "Engineering", "verified": True},
    "hr@acmecorp.com": {"name": "HR Department", "role": "HR", "department": "Human Resources", "verified": True},
    "legal@acmecorp.com": {"name": "Legal Department", "role": "Legal", "department": "Legal", "verified": True},
    "compliance@acmecorp.com": {"name": "Compliance Team", "role": "Compliance", "department": "Legal", "verified": True},
    "it-security@acmecorp.com": {"name": "IT Security", "role": "Security", "department": "IT", "verified": True},
    "finance-alerts@acmecorp.com": {"name": "Finance Alerts", "role": "Finance", "department": "Finance", "verified": True},
    "dev-team@acmecorp.com": {"name": "Dev Team", "role": "Engineering", "department": "Engineering", "verified": True},
    "support@it-helpdesk.acmecorp.com": {"name": "IT Helpdesk", "role": "IT Support", "department": "IT", "verified": True},
    "alice.johnson@acmecorp.com": {"name": "Alice Johnson", "role": "Software Engineer", "department": "Engineering", "verified": True},
    "marketing@acmecorp.com": {"name": "Marketing Team", "role": "Marketing", "department": "Marketing", "verified": True},
}

KNOWN_CORPORATE_DOMAINS: List[str] = [
    "acmecorp.com",
    "it-helpdesk.acmecorp.com",
]

SUSPICIOUS_DOMAINS: List[str] = [
    "acme-corp-security.net",
    "acmecorp-communications.net",
    "microsoft-security-alert.info",
    "microsofT-login.info",
    "suspicious-domain.xyz",
    "deals4u-promo.biz",
    "shopnow-weekly.com",
    "industry-recognition-committee.org",
]


# ─────────────────────────────────────────────
# Tool Implementations
# ─────────────────────────────────────────────

def lookup_sender(email_address: str) -> Dict[str, Any]:
    """Look up sender in corporate directory."""
    if email_address in CORPORATE_DIRECTORY:
        info = CORPORATE_DIRECTORY[email_address]
        return {
            "found": True,
            "email": email_address,
            "name": info["name"],
            "role": info["role"],
            "department": info["department"],
            "verified": info["verified"],
        }
    return {
        "found": False,
        "email": email_address,
        "message": "Sender not found in corporate directory.",
        "verified": False,
    }


def analyze_links(links: List[str]) -> Dict[str, Any]:
    """Analyze URLs for suspicious patterns."""
    results = []
    for link in links:
        suspicious = False
        reasons = []

        # Check for suspicious TLDs
        suspicious_tlds = [".xyz", ".biz", ".info", ".net"]
        for tld in suspicious_tlds:
            if tld in link:
                suspicious = True
                reasons.append(f"Suspicious TLD ({tld}) detected")

        # Check for homograph attacks (mixed case in domain)
        domain_part = link.split("/")[2] if "://" in link else link
        if any(c.isupper() for c in domain_part):
            suspicious = True
            reasons.append("Mixed case in domain name (possible homograph attack)")

        # Check for known suspicious domains
        for bad_domain in SUSPICIOUS_DOMAINS:
            if bad_domain in link:
                suspicious = True
                reasons.append(f"Known suspicious domain: {bad_domain}")

        # Check for IP addresses in URL
        import re
        if re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", link):
            suspicious = True
            reasons.append("IP address used instead of domain name")

        results.append({
            "url": link,
            "suspicious": suspicious,
            "reasons": reasons,
        })

    overall_risk = "high" if any(r["suspicious"] for r in results) else "low"
    return {
        "links_analyzed": len(links),
        "overall_risk": overall_risk,
        "details": results,
    }


def check_sender_domain(email_address: str) -> Dict[str, Any]:
    """Verify sender domain against known corporate domains."""
    try:
        domain = email_address.split("@")[1]
    except IndexError:
        return {"error": "Invalid email address format"}

    is_corporate = domain in KNOWN_CORPORATE_DOMAINS
    is_suspicious = domain in SUSPICIOUS_DOMAINS

    return {
        "email": email_address,
        "domain": domain,
        "is_corporate_domain": is_corporate,
        "is_suspicious_domain": is_suspicious,
        "risk_level": "high" if is_suspicious else ("low" if is_corporate else "medium"),
        "recommendation": (
            "Block and report to security" if is_suspicious
            else ("Trusted internal sender" if is_corporate
                  else "Verify sender identity before acting")
        ),
    }


def search_email_history(email_address: str, limit: int = 5) -> Dict[str, Any]:
    """Search mock email history for communication patterns."""
    history_db: Dict[str, List[Dict[str, Any]]] = {
        "ceo@acmecorp.com": [
            {"subject": "Q3 Results Review", "date": "2024-01-10", "direction": "received"},
            {"subject": "All Hands Meeting", "date": "2024-01-05", "direction": "received"},
        ],
        "cfo@acmecorp.com": [
            {"subject": "Q3 Budget Review", "date": "2024-01-08", "direction": "received"},
        ],
        "securityalert@acme-corp-security.net": [],
        "noreply@microsoft-security-alert.info": [],
        "ceo@acmecorp-communications.net": [],
    }

    history = history_db.get(email_address, [])
    return {
        "email": email_address,
        "history_count": len(history),
        "previous_communications": history[:limit],
        "first_contact": len(history) == 0,
        "note": "No prior communication history — exercise caution" if len(history) == 0 else None,
    }


def flag_security_incident(email_id: str, reason: str) -> Dict[str, Any]:
    """Flag an email as a security incident."""
    return {
        "success": True,
        "email_id": email_id,
        "incident_id": f"INC-{hash(email_id) % 100000:05d}",
        "reason": reason,
        "status": "Reported to Security Operations Center",
        "action_taken": "Email quarantined and security team notified",
    }


# ─────────────────────────────────────────────
# Tool Dispatcher
# ─────────────────────────────────────────────

def execute_tool(tool_name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
    """Route tool call to the appropriate implementation."""
    tools_map = {
        "lookup_sender": lambda args: lookup_sender(args.get("email_address", "")),
        "analyze_links": lambda args: analyze_links(args.get("links", [])),
        "check_sender_domain": lambda args: check_sender_domain(args.get("email_address", "")),
        "search_email_history": lambda args: search_email_history(
            args.get("email_address", ""), args.get("limit", 5)
        ),
        "flag_security_incident": lambda args: flag_security_incident(
            args.get("email_id", ""), args.get("reason", "Suspicious email")
        ),
    }

    if tool_name not in tools_map:
        return {"error": f"Unknown tool: '{tool_name}'. Available: {list(tools_map.keys())}"}

    try:
        return tools_map[tool_name](tool_args)
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}
