"""
inference.py — Run an LLM agent against the Email Triage OpenEnv.

Environment variables:
  API_BASE_URL  — LLM API base URL (e.g., https://api.groq.com/openai/v1)
  MODEL_NAME    — Model identifier (e.g., llama3-70b-8192)
  HF_TOKEN      — Hugging Face token (used as API key)

Log format (strict):
  [START] task=<task> env=<env_name> model=<model>
  [STEP]  step=<n> action=<string> reward=<float> done=<true/false> error=<null/string>
  [END]   success=<true/false> steps=<n> rewards=<comma-separated>
"""

from __future__ import annotations
import os
import sys
import json
import time
import httpx
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama3-70b-8192")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
TASK_ID = os.environ.get("TASK_ID", "medium")
ENV_NAME = "advanced-enterprise-email-triage"
MAX_STEPS = 30
REQUEST_TIMEOUT = 30  # seconds

# ─────────────────────────────────────────────
# OpenAI Client (Groq-compatible)
# ─────────────────────────────────────────────

client = OpenAI(
    api_key=HF_TOKEN or "dummy-key",
    base_url=API_BASE_URL,
    timeout=REQUEST_TIMEOUT,
)

# ─────────────────────────────────────────────
# Environment HTTP Client
# ─────────────────────────────────────────────

http_client = httpx.Client(timeout=REQUEST_TIMEOUT)


def reset_env(task_id: str) -> Dict[str, Any]:
    resp = http_client.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()


def step_env(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = http_client.post(f"{ENV_URL}/step", json=action)
    resp.raise_for_status()
    return resp.json()


def get_state() -> Dict[str, Any]:
    resp = http_client.get(f"{ENV_URL}/state")
    resp.raise_for_status()
    return resp.json()


# ─────────────────────────────────────────────
# System Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an enterprise email triage specialist. Your job is to process emails in the inbox.

For each email you must determine:
1. PRIORITY: urgent | high | medium | low
2. CATEGORY: phishing | urgent_business | internal_task | marketing | hr | legal | it_support | finance | spam
3. ROUTE_TO: security | executive | manager | it | hr | finance | archive | trash

Important rules:
- Phishing emails (fake security alerts, suspicious domains, social engineering) → category=phishing, route_to=security
- CEO wire transfer requests with urgency/secrecy from unknown addresses → likely phishing
- Emails from unrecognized domains claiming to be internal → phishing
- Legitimate urgent business matters → urgent_business, route_to=executive or manager
- HR communications → hr, route_to=hr
- Finance/budget items → finance, route_to=finance
- IT issues → it_support, route_to=it
- Legal matters → legal, route_to=executive or manager
- Marketing/promotional emails → marketing or spam, route_to=trash or archive
- Internal team communications → internal_task, route_to=manager

You must respond with a valid JSON action. Available action types:
- triage: {"action_type":"triage","email_id":"<id>","priority":"<p>","category":"<c>","route_to":"<r>"}
- use_tool: {"action_type":"use_tool","tool_name":"<name>","tool_args":{...}}
- done: {"action_type":"done"}

Triage all emails before calling done. Be systematic and careful about phishing detection."""


# ─────────────────────────────────────────────
# Build User Message from Observation
# ─────────────────────────────────────────────

def build_user_message(observation: Dict[str, Any]) -> str:
    inbox = observation.get("inbox", [])
    triaged = observation.get("triaged", [])
    step = observation.get("step_count", 0)
    max_steps = observation.get("max_steps", 30)
    message = observation.get("message", "")

    lines = [f"Step {step}/{max_steps}. {message}", ""]

    if inbox:
        lines.append(f"INBOX ({len(inbox)} emails):")
        for email in inbox:
            lines.append(f"""
Email ID: {email['id']}
From: {email['sender']}
Subject: {email['subject']}
Time: {email['timestamp']}
Body: {email['body']}
---""")
    else:
        lines.append("INBOX: Empty (all emails processed)")

    if triaged:
        lines.append(f"\nALREADY TRIAGED ({len(triaged)}):")
        for t in triaged:
            lines.append(f"  - {t['email_id']}: {t['category']} / {t['priority']} → {t['route_to']}")

    if not inbox:
        lines.append("\nAll emails processed. Call done action now.")
    else:
        lines.append(f"\nProcess the next email or call done if finished.")

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Parse Agent Response to Action
# ─────────────────────────────────────────────

def parse_action(response_text: str) -> Dict[str, Any]:
    """Extract JSON action from LLM response."""
    text = response_text.strip()

    # Try to extract JSON block
    import re
    json_match = re.search(r'\{.*?\}', text, re.DOTALL)
    if json_match:
        try:
            action = json.loads(json_match.group())
            return action
        except json.JSONDecodeError:
            pass

    # Try parsing the whole response as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Fallback: done action
    return {"action_type": "done"}


# ─────────────────────────────────────────────
# Main Inference Loop
# ─────────────────────────────────────────────

def run_inference(task_id: str = TASK_ID) -> None:
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")

    # Reset environment
    try:
        observation = reset_env(task_id)
    except Exception as e:
        print(f"[END] success=false steps=0 rewards=")
        sys.exit(1)

    conversation: List[Dict[str, str]] = []
    rewards: List[float] = []
    step = 0
    done = False

    while not done and step < MAX_STEPS:
        step += 1
        user_message = build_user_message(observation)
        conversation.append({"role": "user", "content": user_message})

        # Get LLM action with timeout protection
        action_str = ""
        error_str = None
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation,
                max_tokens=512,
                temperature=0.0,
            )
            action_str = completion.choices[0].message.content or ""
            conversation.append({"role": "assistant", "content": action_str})
        except Exception as e:
            error_str = str(e)
            action_str = json.dumps({"action_type": "done"})
            conversation.append({"role": "assistant", "content": action_str})

        # Parse action
        action = parse_action(action_str)

        # Step environment
        reward = 0.0
        try:
            step_result = step_env(action)
            observation = step_result["observation"]
            reward = float(step_result["reward"])
            done = bool(step_result["done"])
            rewards.append(reward)
        except Exception as e:
            error_str = error_str or str(e)
            done = True

        # Log step
        action_display = action.get("action_type", "unknown")
        if action.get("email_id"):
            action_display += f":{action['email_id']}"
        # Sanitize action_display and error_str — no spaces/newlines in log fields
        action_display = action_display.replace(" ", "_").replace("\n", "")
        error_log = "null" if error_str is None else error_str.replace("\n", " ").replace("\r", "")[:120]

        print(
            f"[STEP] step={step} "
            f"action={action_display} "
            f"reward={reward:.4f} "
            f"done={'true' if done else 'false'} "
            f"error={error_log}"
        )

        # Force done if inbox is empty and agent hasn't called done
        if observation.get("inbox") == [] and not done:
            # Give agent one more step to call done
            if action.get("action_type") != "done":
                try:
                    step_result = step_env({"action_type": "done"})
                    reward = float(step_result["reward"])
                    done = True
                    rewards.append(reward)
                    step += 1
                    print(
                        f"[STEP] step={step} action=done "
                        f"reward={reward:.4f} done=true error=null"
                    )
                except Exception:
                    done = True

    rewards_str = ",".join(f"{r:.4f}" for r in rewards)
    success = len(rewards) > 0 and rewards[-1] > 0.0

    print(f"[END] success={'true' if success else 'false'} steps={step} rewards={rewards_str}")


# ─────────────────────────────────────────────
# Entry
# ─────────────────────────────────────────────

if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else TASK_ID
    run_inference(task_id=task)
