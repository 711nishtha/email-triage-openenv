from __future__ import annotations
import os
import sys
import json
import re
import httpx
from typing import Any, Dict, List, Optional
from openai import OpenAI

# ─────────────────────────────────────────────
# Environment variables – safe access
# ─────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("GROQ_API_KEY")  # fallback

# Check required variables
missing = []
if not API_BASE_URL:
    missing.append("API_BASE_URL")
if not MODEL_NAME:
    missing.append("MODEL_NAME")
if not HF_TOKEN:
    missing.append("HF_TOKEN (or GROQ_API_KEY)")

if missing:
    print(f"[START] task=unknown env=advanced-enterprise-email-triage model=unknown")
    print(f"[END] success=false steps=0 rewards=")
    print(f"ERROR: Missing environment variables: {', '.join(missing)}", file=sys.stderr)
    sys.exit(1)

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
TASK_ID = os.getenv("TASK_ID", "medium")
MAX_STEPS = 30

# ─────────────────────────────────────────────
# OpenAI client (Groq-compatible)
# ─────────────────────────────────────────────

client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
    timeout=30.0,
)

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
# Prompt & action parsing
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
    # Try to extract JSON from the LLM response
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            action = json.loads(match.group())
            # Ensure required fields are present
            if action.get("action_type") == "triage":
                # Fill missing optional fields with defaults
                action.setdefault("priority", "medium")
                action.setdefault("category", "internal_task")
                action.setdefault("route_to", "manager")
            return action
        except json.JSONDecodeError:
            pass
    # Fallback: if there are emails left, triage the first one
    return {"action_type": "done"}

# ─────────────────────────────────────────────
# Main loop with exact log format
# ─────────────────────────────────────────────

def run_inference(task_id: str = TASK_ID) -> None:
    # [START] line – exactly as required
    print(f"[START] task={task_id} env=advanced-enterprise-email-triage model={MODEL_NAME}")

    try:
        observation = reset_env(task_id)
    except Exception as e:
        print(f"[END] success=false steps=0 rewards=")
        print(f"ERROR: Reset failed: {e}", file=sys.stderr)
        return

    rewards: List[float] = []
    step = 0
    done = False
    error_str = None

    while not done and step < MAX_STEPS:
        step += 1
        user_message = build_user_message(observation)

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            action_text = completion.choices[0].message.content or ""
            action = parse_action(action_text)
            error_str = None
        except Exception as e:
            action = {"action_type": "done"}
            error_str = str(e)

        # Execute step
        try:
            result = step_env(action)
            observation = result["observation"]
            reward = float(result["reward"])
            done = bool(result["done"])
            rewards.append(reward)
        except Exception as e:
            error_str = str(e)
            done = True
            reward = 0.0

        # [STEP] line – action is a simple string (action_type)
        action_display = action.get("action_type", "unknown")
        error_log = "null" if error_str is None else error_str[:100]
        print(f"[STEP] step={step} action={action_display} reward={reward:.4f} done={str(done).lower()} error={error_log}")

    # [END] line – success=true if any reward > 0, steps, rewards comma-separated
    success = any(r > 0.0 for r in rewards)
    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={step} rewards={rewards_str}")

if __name__ == "__main__":
    run_inference()
