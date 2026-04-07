from __future__ import annotations
import os
import sys
import json
import httpx
from typing import Any, Dict, List

from openai import OpenAI

# ─────────────────────────────────────────────
# Config (STRICT — NO FALLBACKS FOR LLM)
# ─────────────────────────────────────────────

API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.environ["MODEL_NAME"]
API_KEY = os.environ["API_KEY"]

ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")
TASK_ID = os.environ.get("TASK_ID", "medium")
ENV_NAME = "advanced-enterprise-email-triage"
MAX_STEPS = 30
REQUEST_TIMEOUT = 30

# ─────────────────────────────────────────────
# OpenAI Client (MANDATORY PROXY USAGE)
# ─────────────────────────────────────────────

client = OpenAI(
    api_key=API_KEY,
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


# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an enterprise email triage specialist.

Classify each email with:
- priority
- category
- route_to

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
    import re
    match = re.search(r'\{.*?\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except:
            pass
    return {"action_type": "done"}


# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────

def run_inference(task_id: str = TASK_ID) -> None:
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")

    try:
        observation = reset_env(task_id)
    except:
        print("[END] success=false steps=0 score=0.00 rewards=")
        return

    rewards: List[float] = []
    step = 0
    done = False

    while not done and step < MAX_STEPS:
        step += 1

        user_message = build_user_message(observation)

        error_str = None

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
            action_str = completion.choices[0].message.content or ""
        except Exception as e:
            action_str = json.dumps({"action_type": "done"})
            error_str = str(e)

        action = parse_action(action_str)

        reward = 0.0

        try:
            result = step_env(action)
            observation = result["observation"]
            reward = float(result["reward"])
            done = bool(result["done"])
            rewards.append(reward)
        except Exception as e:
            error_str = str(e)
            done = True

        action_display = action.get("action_type", "unknown")
        error_log = "null" if error_str is None else error_str[:100]

        print(
            f"[STEP] step={step} "
            f"action={action_display} "
            f"reward={reward:.2f} "
            f"done={'true' if done else 'false'} "
            f"error={error_log}"
        )

    # ───── FINAL LOG (FIXED) ─────

    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    score = sum(rewards) / max(len(rewards), 1)
    score = min(max(score, 0.0), 1.0)

    success = score > 0.0

    print(
        f"[END] success={'true' if success else 'false'} "
        f"steps={step} score={score:.2f} rewards={rewards_str}"
    )


if __name__ == "__main__":
    run_inference()
