"""30 task definitions for the Email Triage RL Environment (Easy × 10, Medium × 10, Hard × 10)."""

from __future__ import annotations

from models import (
    Difficulty,
    EmailThread,
    StateCheck,
    StepCriteria,
    SuccessCriteria,
    Task,
    TaskID,
)


def _e(sender: str, subject: str, body: str, ts: str = "2024-01-15 09:00") -> EmailThread:
    return EmailThread(sender=sender, subject=subject, body=body, timestamp=ts)


# ===========================================================================
# EASY — 10 tasks  (action_match grading)
# ===========================================================================

EASY_TASKS: list[Task] = [
    # 0
    Task(
        task_id=TaskID(0),
        difficulty=Difficulty.EASY,
        description="Classify the priority and category of this invoice-dispute email.",
        email=_e(
            "client@bigcorp.com",
            "Invoice #4892 — Incorrect Amount Charged",
            "We were charged $4,200 for March but our contract states $3,800. Please correct immediately.",
        ),
        success_criteria=SuccessCriteria(
            action_match=StepCriteria(action="classify", params={"priority": "high", "category": "billing"})
        ),
        hint_level_1="You need to classify this email by priority and category.",
        hint_level_2="Billing disputes from clients are high priority.",
        hint_level_3='{"action": "classify", "priority": "high", "category": "billing"}',
    ),
    # 1
    Task(
        task_id=TaskID(1),
        difficulty=Difficulty.EASY,
        description="Route this IT helpdesk VPN request to the correct queue.",
        email=_e(
            "employee@company.com",
            "Cannot Connect to VPN",
            "VPN has been down since this morning. Tried rebooting but still failing.",
        ),
        success_criteria=SuccessCriteria(
            action_match=StepCriteria(action="assign_queue", params={"queue": "it"})
        ),
        hint_level_1="Assign this email to a queue.",
        hint_level_2="VPN issues belong in the IT support queue.",
        hint_level_3='{"action": "assign_queue", "queue": "it"}',
    ),
    # 2
    Task(
        task_id=TaskID(2),
        difficulty=Difficulty.EASY,
        description="This email is a phishing attempt. Flag it appropriately.",
        email=_e(
            "security-alert@paypa1-support.com",
            "Urgent: Your Account Has Been Suspended",
            "Your PayPal account is suspended. Click here: http://paypa1-login.ru/verify",
        ),
        success_criteria=SuccessCriteria(
            action_match=StepCriteria(action="flag_phishing"),
            phishing_task=True,
        ),
        hint_level_1="Examine the sender domain very carefully.",
        hint_level_2="'paypa1-support.com' is a lookalike domain — this is phishing.",
        hint_level_3='{"action": "flag_phishing"}',
    ),
    # 3
    Task(
        task_id=TaskID(3),
        difficulty=Difficulty.EASY,
        description="Classify this low-urgency vendor newsletter.",
        email=_e(
            "newsletter@vendor.com",
            "Our Q1 Product Updates",
            "Excited to share our Q1 roadmap. No action required.",
        ),
        success_criteria=SuccessCriteria(
            action_match=StepCriteria(action="classify", params={"priority": "low", "category": "general"})
        ),
        hint_level_1="Classify by priority and category.",
        hint_level_2="Newsletters with no required action are low priority general emails.",
        hint_level_3='{"action": "classify", "priority": "low", "category": "general"}',
    ),
    # 4
    Task(
        task_id=TaskID(4),
        difficulty=Difficulty.EASY,
        description="Route this inbound sales lead to the correct team queue.",
        email=_e(
            "prospect@startup.io",
            "Interested in Enterprise Plan",
            "We are a 200-person startup interested in an enterprise demo.",
        ),
        success_criteria=SuccessCriteria(
            action_match=StepCriteria(action="assign_queue", params={"queue": "sales"})
        ),
        hint_level_1="Assign this email to a queue.",
        hint_level_2="Demo requests from prospects go to the sales queue.",
        hint_level_3='{"action": "assign_queue", "queue": "sales"}',
    ),
    # 5
    Task(
        task_id=TaskID(5),
        difficulty=Difficulty.EASY,
        description="Classify this critical outage report from a key customer.",
        email=_e(
            "cto@enterprise-client.com",
            "CRITICAL: Your API is DOWN",
            "Your API has been returning 503 errors for 30 minutes. We are losing revenue. P1.",
        ),
        success_criteria=SuccessCriteria(
            action_match=StepCriteria(action="classify", params={"priority": "urgent", "category": "support"})
        ),
        hint_level_1="Classify priority and category.",
        hint_level_2="API outages reported by customers are urgent support issues.",
        hint_level_3='{"action": "classify", "priority": "urgent", "category": "support"}',
    ),
    # 6
    Task(
        task_id=TaskID(6),
        difficulty=Difficulty.EASY,
        description="Route this employee parental-leave HR question.",
        email=_e(
            "employee2@company.com",
            "Question About Parental Leave Policy",
            "Could you clarify how many weeks of parental leave we are entitled to?",
        ),
        success_criteria=SuccessCriteria(
            action_match=StepCriteria(action="assign_queue", params={"queue": "hr"})
        ),
        hint_level_1="Route this to the right team.",
        hint_level_2="Parental leave questions belong in the HR queue.",
        hint_level_3='{"action": "assign_queue", "queue": "hr"}',
    ),
    # 7
    Task(
        task_id=TaskID(7),
        difficulty=Difficulty.EASY,
        description="Flag this CEO-fraud / wire-transfer phishing attempt.",
        email=_e(
            "ceo.john@company-secure-msg.com",
            "Confidential Wire Transfer Needed",
            "This is CEO John. Please wire $50,000 to a new vendor immediately. Do not tell anyone.",
        ),
        success_criteria=SuccessCriteria(
            action_match=StepCriteria(action="flag_phishing"),
            phishing_task=True,
        ),
        hint_level_1="Check the sender domain — is it your real company domain?",
        hint_level_2="CEO-fraud uses lookalike domains and urgency. Flag as phishing.",
        hint_level_3='{"action": "flag_phishing"}',
    ),
    # 8
    Task(
        task_id=TaskID(8),
        difficulty=Difficulty.EASY,
        description="Classify this refund request from a customer.",
        email=_e(
            "customer@gmail.com",
            "Refund Request — Order #9901",
            "I ordered the wrong plan and want a refund for the unused month.",
        ),
        success_criteria=SuccessCriteria(
            action_match=StepCriteria(action="classify", params={"priority": "medium", "category": "billing"})
        ),
        hint_level_1="Classify priority and category.",
        hint_level_2="Refund requests are medium priority billing emails.",
        hint_level_3='{"action": "classify", "priority": "medium", "category": "billing"}',
    ),
    # 9
    Task(
        task_id=TaskID(9),
        difficulty=Difficulty.EASY,
        description="Classify this internal mandatory password-reset notification.",
        email=_e(
            "security@company.com",
            "Mandatory Password Reset Required",
            "All employees must reset their passwords by Friday via the self-service portal.",
        ),
        success_criteria=SuccessCriteria(
            action_match=StepCriteria(action="classify", params={"priority": "medium", "category": "it"})
        ),
        hint_level_1="Classify this IT security notification.",
        hint_level_2="Internal IT security policies are medium priority IT emails.",
        hint_level_3='{"action": "classify", "priority": "medium", "category": "it"}',
    ),
]


# ===========================================================================
# MEDIUM — 10 tasks  (multi_step grading, steps len >= 2)
# ===========================================================================

MEDIUM_TASKS: list[Task] = [
    # 10
    Task(
        task_id=TaskID(10),
        difficulty=Difficulty.MEDIUM,
        description="Follow-up on existing ticket — check CRM then route to support.",
        email=_e(
            "customer@acme.com",
            "Re: Ticket #TK-1042 Still Not Resolved",
            "It has been 3 days since I reported this. Ticket #TK-1042 still open.",
            ts="2024-01-18 14:00",
        ),
        thread_history=[
            _e("customer@acme.com", "Bug: Export broken", "Export button broken since last update.", ts="2024-01-15 10:00"),
        ],
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="use_tool", params={"tool": "crm"}),
                StepCriteria(action="assign_queue", params={"queue": "support"}),
            ]
        ),
        hint_level_1="Use a tool first, then route the email.",
        hint_level_2="Look up the ticket in the CRM, then assign to the support queue.",
        hint_level_3='Step 1: {"action":"use_tool","tool":"crm","params":{"ticket":"TK-1042"}} '
                     'Step 2: {"action":"assign_queue","queue":"support"}',
    ),
    # 11
    Task(
        task_id=TaskID(11),
        difficulty=Difficulty.MEDIUM,
        description="Schedule a demo for this sales prospect and send a confirmation reply.",
        email=_e(
            "vp@prospect-corp.com",
            "Ready to See a Demo",
            "We are ready to move forward. Can you book a 45-minute demo for next week?",
        ),
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="use_tool", params={"tool": "calendar"}),
                StepCriteria(action="reply", params={"tone": "professional"}),
            ]
        ),
        hint_level_1="Use a tool and then reply.",
        hint_level_2="Check calendar availability, then send a professional reply.",
        hint_level_3='Step 1: {"action":"use_tool","tool":"calendar","params":{"search":"demo slot"}} '
                     'Step 2: {"action":"reply","tone":"professional","summary":"demo scheduled"}',
    ),
    # 12
    Task(
        task_id=TaskID(12),
        difficulty=Difficulty.MEDIUM,
        description="Create a bug-report ticket and send an empathetic acknowledgment reply.",
        email=_e(
            "user@customer.com",
            "Login Page Crashes on Mobile",
            "Every time I try to log in on iPhone the app crashes. Running iOS 17.",
        ),
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="use_tool", params={"tool": "ticketing"}),
                StepCriteria(action="reply", params={"tone": "empathetic"}),
            ]
        ),
        hint_level_1="Create a ticket and acknowledge the user.",
        hint_level_2="Use the ticketing tool, then send an empathetic reply.",
        hint_level_3='Step 1: {"action":"use_tool","tool":"ticketing","params":{"type":"bug"}} '
                     'Step 2: {"action":"reply","tone":"empathetic","summary":"ticket created"}',
    ),
    # 13
    Task(
        task_id=TaskID(13),
        difficulty=Difficulty.MEDIUM,
        description="Classify this billing dispute then escalate to manager.",
        email=_e(
            "cfo@bigclient.com",
            "Overcharge of $28,000 — Legal Action If Not Resolved",
            "We have been overcharged $28,000. If not resolved in 24 hours we involve legal.",
        ),
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="classify", params={"priority": "urgent"}),
                StepCriteria(action="escalate", params={"to": "manager"}),
            ]
        ),
        hint_level_1="Classify and then escalate.",
        hint_level_2="Large billing disputes with legal threats need urgent classification then manager escalation.",
        hint_level_3='Step 1: {"action":"classify","priority":"urgent","category":"billing"} '
                     'Step 2: {"action":"escalate","to":"manager","reason":"legal threat"}',
    ),
    # 14
    Task(
        task_id=TaskID(14),
        difficulty=Difficulty.MEDIUM,
        description="Thread-based sales follow-up — check CRM history then route to sales.",
        email=_e(
            "buyer@retailer.com",
            "Re: Pricing Discussion",
            "Following up on our call — ready to sign if you can match $499/mo.",
            ts="2024-01-22 11:00",
        ),
        thread_history=[
            _e("buyer@retailer.com", "Interested in Your Platform", "Can you send pricing?", ts="2024-01-15 09:00"),
            _e("sales@company.com", "Re: Interested in Your Platform", "Standard plan is $599/mo.", ts="2024-01-17 10:00"),
        ],
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="use_tool", params={"tool": "crm"}),
                StepCriteria(action="assign_queue", params={"queue": "sales"}),
            ]
        ),
        hint_level_1="Look up context and route to the right team.",
        hint_level_2="Check CRM for account history, then route to the sales queue.",
        hint_level_3='Step 1: {"action":"use_tool","tool":"crm","params":{"search":"retailer account"}} '
                     'Step 2: {"action":"assign_queue","queue":"sales"}',
    ),
    # 15
    Task(
        task_id=TaskID(15),
        difficulty=Difficulty.MEDIUM,
        description="Create an access-request ticket for this IT request then route to IT queue.",
        email=_e(
            "dev@company.com",
            "Need Admin Rights for CI/CD Setup",
            "I need temporary admin access to configure our new Jenkins server. Manager approved.",
        ),
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="use_tool", params={"tool": "ticketing"}),
                StepCriteria(action="assign_queue", params={"queue": "it"}),
            ]
        ),
        hint_level_1="Create a ticket and route it.",
        hint_level_2="Access requests need a ticket and go to the IT queue.",
        hint_level_3='Step 1: {"action":"use_tool","tool":"ticketing","params":{"type":"access_request"}} '
                     'Step 2: {"action":"assign_queue","queue":"it"}',
    ),
    # 16
    Task(
        task_id=TaskID(16),
        difficulty=Difficulty.MEDIUM,
        description="Log this new customer in CRM then send a welcoming onboarding reply.",
        email=_e(
            "admin@new-customer.com",
            "Just Signed Up — Need Onboarding Help",
            "We just completed our contract. Looking forward to getting started!",
        ),
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="use_tool", params={"tool": "crm"}),
                StepCriteria(action="reply", params={"tone": "welcoming"}),
            ]
        ),
        hint_level_1="Update CRM and send a reply.",
        hint_level_2="Add to CRM first, then send a welcoming reply to the new customer.",
        hint_level_3='Step 1: {"action":"use_tool","tool":"crm","params":{"action":"create_account"}} '
                     'Step 2: {"action":"reply","tone":"welcoming","summary":"onboarding started"}',
    ),
    # 17
    Task(
        task_id=TaskID(17),
        difficulty=Difficulty.MEDIUM,
        description="Escalated complaint in thread — classify urgent then escalate to director.",
        email=_e(
            "vp@unhappy-client.com",
            "Re: Unresolved Issues — Escalating to Leadership",
            "Third time reaching out. Escalating to your director level now.",
            ts="2024-01-25 16:00",
        ),
        thread_history=[
            _e("vp@unhappy-client.com", "Critical Performance Issues", "Severe latency on your platform.", ts="2024-01-20 09:00"),
            _e("vp@unhappy-client.com", "Re: Critical Performance Issues", "Still no resolution after 3 days.", ts="2024-01-23 14:00"),
        ],
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="classify", params={"priority": "urgent"}),
                StepCriteria(action="escalate", params={"to": "director"}),
            ]
        ),
        hint_level_1="Classify urgency and escalate.",
        hint_level_2="Repeat escalations from VPs are urgent and should go to director level.",
        hint_level_3='Step 1: {"action":"classify","priority":"urgent","category":"support"} '
                     'Step 2: {"action":"escalate","to":"director","reason":"repeat escalation"}',
    ),
    # 18
    Task(
        task_id=TaskID(18),
        difficulty=Difficulty.MEDIUM,
        description="Route this job application to HR and send an acknowledgment reply.",
        email=_e(
            "applicant@gmail.com",
            "Application for Senior Engineer Role",
            "Please find my resume for the Senior Software Engineer position on LinkedIn.",
        ),
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="assign_queue", params={"queue": "hr"}),
                StepCriteria(action="reply", params={"tone": "professional"}),
            ]
        ),
        hint_level_1="Route and reply.",
        hint_level_2="Job applications go to HR and need an acknowledgment reply.",
        hint_level_3='Step 1: {"action":"assign_queue","queue":"hr"} '
                     'Step 2: {"action":"reply","tone":"professional","summary":"application received"}',
    ),
    # 19
    Task(
        task_id=TaskID(19),
        difficulty=Difficulty.MEDIUM,
        description="Payment failure from Stripe — look up customer in CRM then route to billing.",
        email=_e(
            "noreply@stripe.com",
            "Payment Failed for Customer ID C-8821",
            "Automated notice: payment of $1,200 for C-8821 failed. Card declined.",
        ),
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="use_tool", params={"tool": "crm"}),
                StepCriteria(action="assign_queue", params={"queue": "billing"}),
            ]
        ),
        hint_level_1="Look up the customer and route.",
        hint_level_2="Check CRM for customer details, then send to billing queue.",
        hint_level_3='Step 1: {"action":"use_tool","tool":"crm","params":{"customer_id":"C-8821"}} '
                     'Step 2: {"action":"assign_queue","queue":"billing"}',
    ),
]


# ===========================================================================
# HARD — 10 tasks  (state_checks grading)
# ===========================================================================

HARD_TASKS: list[Task] = [
    # 20 — subtle phishing + state check
    Task(
        task_id=TaskID(20),
        difficulty=Difficulty.HARD,
        description="Detect this subtle IT-password-reset phishing email and flag it.",
        email=_e(
            "it-helpdesk@company-security-portal.net",
            "Action Required: Reset Your Corporate Password",
            "Your password expires in 2 hours. Reset: http://company-security-portal.net/reset?token=abc123",
        ),
        success_criteria=SuccessCriteria(
            steps=[StepCriteria(action="flag_phishing")],
            state_checks=[
                StateCheck(field="flags_raised", expected="phishing", mode="contains"),
            ],
            phishing_task=True,
        ),
        hint_level_1="Examine the sender domain closely.",
        hint_level_2="'company-security-portal.net' is not your corporate domain — it is phishing.",
        hint_level_3='{"action": "flag_phishing"}',
    ),
    # 21 — SLA breach: classify + ticket + escalate + state check
    Task(
        task_id=TaskID(21),
        difficulty=Difficulty.HARD,
        description=(
            "Customer at SLA breach risk (4-hour SLA, 3.5 hours elapsed). "
            "Classify urgent, create ticket, escalate to manager."
        ),
        email=_e(
            "ops@sla-customer.com",
            "Database Issue — 3.5 Hours No Response",
            "P1 database connectivity issue reported 3.5 hours ago. 30 minutes left on your SLA.",
        ),
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="classify", params={"priority": "urgent"}),
                StepCriteria(action="use_tool", params={"tool": "ticketing"}),
                StepCriteria(action="escalate", params={"to": "manager"}),
            ],
            state_checks=[
                StateCheck(field="flags_raised", expected="sla_breach", mode="contains"),
                StateCheck(field="tool_calls", expected="ticketing", mode="contains"),
            ],
        ),
        hint_level_1="Classify, create a ticket, escalate, and flag the SLA breach.",
        hint_level_2="SLA breach: urgent + ticketing tool + manager escalation + raise sla_breach flag.",
        hint_level_3='Step 1: {"action":"classify","priority":"urgent"} '
                     'Step 2: {"action":"use_tool","tool":"ticketing"} '
                     'Step 3: {"action":"escalate","to":"manager","reason":"sla_breach"}',
    ),
    # 22 — churn-risk pipeline: CRM + ticket + notification + state checks
    Task(
        task_id=TaskID(22),
        difficulty=Difficulty.HARD,
        description=(
            "High-value customer ($200k/yr) is churning. Run: CRM lookup → ticket creation → "
            "account-manager notification."
        ),
        email=_e(
            "ceo@whale-account.com",
            "Reconsidering Our Subscription",
            "$200k/yr customer — no contact from account manager in 6 weeks. Evaluating alternatives.",
        ),
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="use_tool", params={"tool": "crm"}),
                StepCriteria(action="use_tool", params={"tool": "ticketing"}),
                StepCriteria(action="use_tool", params={"tool": "notification"}),
            ],
            state_checks=[
                StateCheck(field="tool_calls", expected="crm", mode="contains"),
                StateCheck(field="tool_calls", expected="ticketing", mode="contains"),
                StateCheck(field="tool_calls", expected="notification", mode="contains"),
            ],
        ),
        hint_level_1="This requires three tool calls in sequence.",
        hint_level_2="CRM to identify account → ticket for churn risk → notification to account manager.",
        hint_level_3='Step 1: {"action":"use_tool","tool":"crm"} '
                     'Step 2: {"action":"use_tool","tool":"ticketing"} '
                     'Step 3: {"action":"use_tool","tool":"notification","params":{"to":"account_manager"}}',
    ),
    # 23 — DocuSign phishing + state check
    Task(
        task_id=TaskID(23),
        difficulty=Difficulty.HARD,
        description="Detect credential-harvesting phishing disguised as a DocuSign request.",
        email=_e(
            "dochub@docusign-secure.biz",
            "Document Awaiting Your Signature",
            "Sign here: https://docusign-secure.biz/d/review?id=9912 Expires in 1 hour.",
        ),
        success_criteria=SuccessCriteria(
            steps=[StepCriteria(action="flag_phishing")],
            state_checks=[
                StateCheck(field="flags_raised", expected="phishing", mode="contains"),
            ],
            phishing_task=True,
        ),
        hint_level_1="Check the sender domain vs the real DocuSign domain.",
        hint_level_2="Real DocuSign uses docusign.com — 'docusign-secure.biz' is a lookalike.",
        hint_level_3='{"action": "flag_phishing"}',
    ),
    # 24 — ambiguous email: reply + route + state check
    Task(
        task_id=TaskID(24),
        difficulty=Difficulty.HARD,
        description=(
            "Ambiguous API question from a free-tier user that is a sales upsell opportunity. "
            "Reply professionally then route to sales."
        ),
        email=_e(
            "contact@prospect.com",
            "Question About Your API Rate Limits",
            "We are a free-tier user wondering if API rate limits can be customised.",
        ),
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="reply", params={"tone": "professional"}),
                StepCriteria(action="assign_queue", params={"queue": "sales"}),
            ],
            state_checks=[
                StateCheck(field="actions_taken", expected="reply", mode="contains"),
            ],
        ),
        hint_level_1="Reply to clarify and then route.",
        hint_level_2="Free-tier users asking about API limits are upsell leads — reply then route to sales.",
        hint_level_3='Step 1: {"action":"reply","tone":"professional","summary":"clarifying question"} '
                     'Step 2: {"action":"assign_queue","queue":"sales"}',
    ),
    # 25 — P0 outage: classify + on-call escalate + notify + state checks
    Task(
        task_id=TaskID(25),
        difficulty=Difficulty.HARD,
        description=(
            "Critical P0 production outage with SLA breach. "
            "Classify urgent, escalate to on-call, notify via tool, flag sla_breach."
        ),
        email=_e(
            "sre@enterprise.com",
            "OUTAGE: Complete Service Disruption P0",
            "All API calls failing with 500 errors for 4 hours. 100k users impacted.",
        ),
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="classify", params={"priority": "urgent"}),
                StepCriteria(action="escalate", params={"to": "on-call"}),
                StepCriteria(action="use_tool", params={"tool": "notification"}),
            ],
            state_checks=[
                StateCheck(field="flags_raised", expected="sla_breach", mode="contains"),
                StateCheck(field="tool_calls", expected="notification", mode="contains"),
            ],
        ),
        hint_level_1="Classify, escalate, notify, and flag SLA breach.",
        hint_level_2="P0 outages: urgent + on-call escalation + notification tool + sla_breach flag.",
        hint_level_3='Step 1: {"action":"classify","priority":"urgent"} '
                     'Step 2: {"action":"escalate","to":"on-call","reason":"sla_breach"} '
                     'Step 3: {"action":"use_tool","tool":"notification","params":{"channel":"incident"}}',
    ),
    # 26 — fake AWS invoice spear-phishing + state check
    Task(
        task_id=TaskID(26),
        difficulty=Difficulty.HARD,
        description="Detect invoice-fraud phishing that impersonates AWS and requests a wire transfer.",
        email=_e(
            "accounts@aws-invoices-billing.com",
            "AWS Invoice #INV-2024-003 — Payment Due",
            "Pay your AWS invoice of $12,450 via wire transfer. Discount expires today.",
        ),
        success_criteria=SuccessCriteria(
            steps=[StepCriteria(action="flag_phishing")],
            state_checks=[
                StateCheck(field="flags_raised", expected="phishing", mode="contains"),
            ],
            phishing_task=True,
        ),
        hint_level_1="AWS does not request wire transfers via email.",
        hint_level_2="'aws-invoices-billing.com' is not amazon.com — classic invoice fraud.",
        hint_level_3='{"action": "flag_phishing"}',
    ),
    # 27 — partner integration failure: CRM + ticket + escalate + state checks
    Task(
        task_id=TaskID(27),
        difficulty=Difficulty.HARD,
        description=(
            "Partner webhook integration failure blocking their payment pipeline. "
            "CRM lookup, create ticket, escalate to manager."
        ),
        email=_e(
            "tech@partner-api.com",
            "Our Webhook Integration is Failing",
            "Since your last deploy, our webhook endpoint receives no events. Blocking payment processing.",
        ),
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="use_tool", params={"tool": "crm"}),
                StepCriteria(action="use_tool", params={"tool": "ticketing"}),
                StepCriteria(action="escalate", params={"to": "manager"}),
            ],
            state_checks=[
                StateCheck(field="tool_calls", expected="crm", mode="contains"),
                StateCheck(field="tool_calls", expected="ticketing", mode="contains"),
            ],
        ),
        hint_level_1="Lookup, create ticket, and escalate.",
        hint_level_2="Partner integration failures: CRM lookup + ticket + manager escalation.",
        hint_level_3='Step 1: {"action":"use_tool","tool":"crm"} '
                     'Step 2: {"action":"use_tool","tool":"ticketing","params":{"severity":"high"}} '
                     'Step 3: {"action":"escalate","to":"manager","reason":"partner_integration_failure"}',
    ),
    # 28 — internal credential-harvesting phishing + state check
    Task(
        task_id=TaskID(28),
        difficulty=Difficulty.HARD,
        description=(
            "Detect sophisticated internal phishing: email claims to be from IT "
            "but asks for your password from an external domain."
        ),
        email=_e(
            "it-admin@company-helpdesk-support.org",
            "IT Security Audit — Submit Credentials",
            "Part of our security audit — please reply with your username and password.",
        ),
        success_criteria=SuccessCriteria(
            steps=[StepCriteria(action="flag_phishing")],
            state_checks=[
                StateCheck(field="flags_raised", expected="phishing", mode="contains"),
            ],
            phishing_task=True,
        ),
        hint_level_1="Legitimate IT never asks for your password via email.",
        hint_level_2="Credential requests via email are always phishing — regardless of sender name.",
        hint_level_3='{"action": "flag_phishing"}',
    ),
    # 29 — VIP missed SLA with thread: classify + CRM + VP escalate + notify + state checks
    Task(
        task_id=TaskID(29),
        difficulty=Difficulty.HARD,
        description=(
            "VIP enterprise customer with missed SLA — 2-week-old feature request with no response. "
            "Classify urgent, log in CRM, escalate to VP, notify customer."
        ),
        email=_e(
            "cpo@vip-enterprise.com",
            "Re: Feature Request — No Response in 2 Weeks",
            "Escalated to your VP 2 weeks ago. Board meeting tomorrow. Still no response.",
            ts="2024-01-29 08:00",
        ),
        thread_history=[
            _e("cpo@vip-enterprise.com", "Critical Feature for Q1 Launch", "Need bulk export for Q1 board presentation.", ts="2024-01-10 09:00"),
            _e("cpo@vip-enterprise.com", "Re: Critical Feature for Q1 Launch", "Still no reply on bulk export.", ts="2024-01-15 14:00"),
        ],
        success_criteria=SuccessCriteria(
            steps=[
                StepCriteria(action="classify", params={"priority": "urgent"}),
                StepCriteria(action="use_tool", params={"tool": "crm"}),
                StepCriteria(action="escalate", params={"to": "vp"}),
                StepCriteria(action="use_tool", params={"tool": "notification"}),
            ],
            state_checks=[
                StateCheck(field="tool_calls", expected="crm", mode="contains"),
                StateCheck(field="tool_calls", expected="notification", mode="contains"),
                StateCheck(field="flags_raised", expected="sla_breach", mode="contains"),
            ],
        ),
        hint_level_1="Classify, use tools, escalate, and notify.",
        hint_level_2="VIP missed SLA: urgent + CRM log + VP escalation + notification + sla_breach flag.",
        hint_level_3='Step 1: {"action":"classify","priority":"urgent"} '
                     'Step 2: {"action":"use_tool","tool":"crm"} '
                     'Step 3: {"action":"escalate","to":"vp","reason":"sla_breach"} '
                     'Step 4: {"action":"use_tool","tool":"notification","params":{"to":"customer"}}',
    ),
]


ALL_TASKS: list[Task] = EASY_TASKS + MEDIUM_TASKS + HARD_TASKS
TASKS_BY_ID: dict[int, Task] = {int(t.task_id): t for t in ALL_TASKS}
