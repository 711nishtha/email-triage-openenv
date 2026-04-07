from __future__ import annotations
import os
import sys
import json
import re
import httpx
from typing import Any, Dict, List, Optional

# ─────────────────────────────────────────────
# Environment variables – safe defaults
# ─────────────────────────────────────────────

# Provide defaults so the script never crashes
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-70b-8192")  # fallback model name
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY", "dummy")  # dummy to avoid None

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
TASK_ID = os.getenv("TASK_ID", "medium")
MAX_STEPS = 30

# ─────────────────────────────────────────────
# OpenAI client – will work if API key is valid,
# but if not, we catch exceptions and use heuristic.
# ─────────────────────────────────────────────
try:
    from openai import OpenAI
    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL, timeout=30.0)
    USE_LLM = True
except Exception:
    USE_LLM = False

http_client = httpx.Client(timeout=30.0)

# ─────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────

def reset_env(task_id: str) -> Dict[str, Any]:
    resp = http_client.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()

def step_env(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = http_client.post(f"{ENV_URL}/step", json=action)
    resp.raise_for_status()
    return resp.json()

# ─────────────────────────────────────────────
# Deterministic heuristic (always works)
# ─────────────────────────────────────────────

def heuristic_action(observation: Dict[str, Any]) -> Dict[str, Any]:
    inbox = observation.get("inbox", [])
    if not inbox:
        return {"action_type": "done"}

    email = inbox[0]
    subject = email.get("subject", "").lower()
    body = email.get("body", "").lower()
    sender = email.get("sender", "").lower()

    # Priority
    if "urgent" in subject or "critical" in subject or "immediate" in body:
        priority = "urgent"
    elif "deadline" in subject or "asap" in body:
        priority = "high"
    else:
        priority = "medium"

    # Category
    if "phish" in body or "verify your account" in body or "click here" in body:
        category = "phishing"
    elif "urgent" in subject or "board" in subject:
        category = "urgent_business"
    elif "hr" in sender or "benefits" in subject:
        category = "hr"
    elif "finance" in sender or "invoice" in subject:
        category = "finance"
    elif "it" in sender or "password" in subject:
        category = "it_support"
    elif "marketing" in sender or "deal" in subject:
        category = "marketing"
    else:
        category = "internal_task"

    # Route
    if category == "phishing":
        route_to = "security"
    elif category == "urgent_business":
        route_to = "executive"
    elif category == "finance":
        route_to = "finance"
    elif category == "it_support":
        route_to = "it"
    elif category == "hr":
        route_to = "hr"
    else:
        route_to = "manager"

    return {
        "action_type": "triage",
        "email_id": email["id"],
        "priority": priority,
        "category": category,
        "route_to": route_to,
    }

# ─────────────────────────────────────────────
# LLM prompt and parsing (if available)
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an enterprise email triage specialist.

Classify each email with:
- priority: urgent, high, medium, low
- category: phishing, urgent_business, internal_task, marketing, hr, legal, it_support, finance, spam
- route_to: security, executive, manager, it, hr, finance, archive, trash

Return ONLY valid JSON:
{"action_type":"triage","email_id":"<id>","priority":"<p>","category":"<c>","route_to":"<r>"}
OR {"action_type":"done"}
"""

def build_user_message(observation: Dict[str, Any]) -> str:
    inbox = observation.get("inbox", [])
    step = observation.get("step_count", 0)
    max_steps = observation.get("max_steps", 30)
    lines = [f"Step {step}/{max_steps}"]
    for email in inbox:
        lines.append(f"""
ID: {email['id']}
From: {email['sender']}
Subject: {email['subject']}
Body: {email['body']}
---""")
    if not inbox:
        lines.append("All emails processed. Call done.")
    return "\n".join(lines)

def parse_action(text: str) -> Dict[str, Any]:
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            action = json.loads(match.group())
            if action.get("action_type") == "triage":
                action.setdefault("priority", "medium")
                action.setdefault("category", "internal_task")
                action.setdefault("route_to", "manager")
            return action
        except json.JSONDecodeError:
            pass
    return {"action_type": "done"}

# ─────────────────────────────────────────────
# Main loop – no crashes, always logs
# ─────────────────────────────────────────────

def run_inference(task_id: str = TASK_ID) -> None:
    # [START] – use fallback model name if missing
    model_display = MODEL_NAME if MODEL_NAME else "fallback-heuristic"
    print(f"[START] task={task_id} env=advanced-enterprise-email-triage model={model_display}")

    try:
        observation = reset_env(task_id)
    except Exception as e:
        # If reset fails, we still need to produce [END] log
        print(f"[END] success=false steps=0 rewards=")
        return

    rewards: List[float] = []
    step = 0
    done = False
    error_str = None

    while not done and step < MAX_STEPS:
        step += 1

        # Choose action: try LLM first, fallback to heuristic
        if USE_LLM:
            try:
                user_msg = build_user_message(observation)
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                )
                action_text = completion.choices[0].message.content or ""
                action = parse_action(action_text)
                error_str = None
            except Exception as e:
                action = heuristic_action(observation)
                error_str = str(e)
        else:
            action = heuristic_action(observation)
            error_str = "LLM disabled (missing or invalid API credentials)"

        # Execute the step
        try:
            result = step_env(action)
            observation = result["observation"]
            raw_reward = float(result["reward"])
            # Clamp reward to [0.0, 1.0] – validator is strict
            reward = max(0.0, min(1.0, raw_reward))
            done = bool(result["done"])
            rewards.append(reward)
        except Exception as e:
            error_str = str(e)
            done = True
            reward = 0.0

        action_display = action.get("action_type", "unknown")
        error_log = "null" if error_str is None else error_str[:100]
        print(f"[STEP] step={step} action={action_display} reward={reward:.4f} done={str(done).lower()} error={error_log}")

    # [END] – success true if any reward > 0
    success = any(r > 0.0 for r in rewards) if rewards else False
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}")

if __name__ == "__main__":
    # Never exit with non-zero – catch all exceptions
    try:
        run_inference()
    except Exception:
        # Absolute last resort: print a valid [END] log
        print("[END] success=false steps=0 rewards=")
