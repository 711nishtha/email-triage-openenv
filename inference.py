from __future__ import annotations
import os
import sys
import json
import httpx
from typing import Any, Dict, List

from openai import OpenAI

# ─────────────────────────────────────────────
# Config (CORRECT HYBRID MODE)
# ─────────────────────────────────────────────

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
API_KEY = os.environ["API_KEY"]  # STRICT

ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")
TASK_ID = os.getenv("TASK_ID", "medium")
ENV_NAME = "advanced-enterprise-email-triage"
MAX_STEPS = 30
REQUEST_TIMEOUT = 30

# ─────────────────────────────────────────────
# OpenAI Client
# ─────────────────────────────────────────────

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
    timeout=REQUEST_TIMEOUT,
)

http_client = httpx.Client(timeout=REQUEST_TIMEOUT)


def reset_env(task_id: str) -> Dict[str, Any]:
    resp = http_client.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    return resp.json()


def step_env(action: Dict[str, Any]) -> Dict[str, Any]:
    resp = http_client.post(f"{ENV_URL}/step", json=action)
    resp.raise_for_status()
    return resp.json()


SYSTEM_PROMPT = """You are an enterprise email triage agent.

Return ONLY JSON:
{"action_type":"triage","email_id":"<id>","priority":"<p>","category":"<c>","route_to":"<r>"}
OR {"action_type":"done"}
"""


def build_user_message(obs: Dict[str, Any]) -> str:
    inbox = obs.get("inbox", [])
    step = obs.get("step_count", 0)
    max_steps = obs.get("max_steps", 30)

    msg = [f"Step {step}/{max_steps}"]

    for e in inbox:
        msg.append(f"""
ID: {e['id']}
From: {e['sender']}
Subject: {e['subject']}
Body: {e['body']}
---""")

    if not inbox:
        msg.append("All emails processed. Call done.")

    return "\n".join(msg)


def parse_action(text: str) -> Dict[str, Any]:
    import re
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except:
            pass
    return {"action_type": "done"}


def run_inference(task_id: str = TASK_ID) -> None:
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")

    try:
        obs = reset_env(task_id)
    except:
        print("[END] success=false steps=0 score=0.00 rewards=")
        return

    rewards: List[float] = []
    step = 0
    done = False

    while not done and step < MAX_STEPS:
        step += 1
        error_str = None

        user_msg = build_user_message(obs)

        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.0,
                max_tokens=300,
            )
            action_str = resp.choices[0].message.content or ""
        except Exception as e:
            action_str = json.dumps({"action_type": "done"})
            error_str = str(e)

        action = parse_action(action_str)

        reward = 0.0
        try:
            result = step_env(action)
            obs = result["observation"]
            reward = float(result["reward"])
            done = bool(result["done"])
            rewards.append(reward)
        except Exception as e:
            error_str = str(e)
            done = True

        error_log = "null" if error_str is None else error_str[:100]

        print(
            f"[STEP] step={step} action={action.get('action_type','unknown')} "
            f"reward={reward:.2f} done={'true' if done else 'false'} error={error_log}"
        )

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
