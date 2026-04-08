"""
Inference Script — Email Triage RL Environment
===============================================

STDOUT FORMAT (required by OpenEnv validator):
  [START] task=<task_id> env=email_triage model=<model>
  [STEP]  step=<n> action=<action_json> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>

Environment variables:
  API_BASE_URL   LLM API endpoint  (e.g. LiteLLM proxy URL)
  API_KEY        API key           (or HF_TOKEN as fallback)
  MODEL_NAME     Model identifier  (default: gpt-4o-mini)
  ENV_URL        Environment URL   (default: http://localhost:7860)
  MAX_STEPS      Max steps/episode (default: 10)
  NUM_EPISODES   Episodes to run   (default: 11)
"""

import os
import textwrap
from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from client import EmailTriageEnv
from models import EmailTriageAction, EmailTriageObservation

load_dotenv()

# ---------------------------------------------------------------------------
# Config — strictly from environment variables
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY: str = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "placeholder"
MODEL_NAME: str = os.getenv("MODEL_NAME") or "gpt-4o-mini"
ENV_URL: str = os.getenv("ENV_URL", "http://localhost:7860")
MAX_STEPS: int = int(os.getenv("MAX_STEPS", "10"))
NUM_EPISODES: int = int(os.getenv("NUM_EPISODES", "11"))
BENCHMARK: str = "email_triage"

# ---------------------------------------------------------------------------
# LLM client (OpenAI-compatible — works with any LiteLLM proxy)
# ---------------------------------------------------------------------------

client_llm = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are an expert enterprise email triage specialist.
    Each turn you must send exactly ONE valid JSON action string.

    Available actions:
      {"action": "classify", "priority": "<low|medium|high|urgent>", "category": "<billing|support|sales|hr|it|general>"}
      {"action": "assign_queue", "queue": "<billing|support|sales|hr|it|general>"}
      {"action": "flag_phishing"}
      {"action": "escalate", "to": "<manager|director|vp|on-call>", "reason": "<reason>"}
      {"action": "reply", "tone": "<professional|empathetic|welcoming>", "summary": "<brief>"}
      {"action": "use_tool", "tool": "<calendar|crm|ticketing|escalation|notification>", "params": {}}

    Rules:
    - Reply with ONLY the JSON string — no explanation, no markdown fences
    - ONE action per turn
    - SAFETY: If the email looks like phishing (lookalike domain, credential request,
      wire transfer, suspicious link), ALWAYS use flag_phishing immediately
    - For escalations involving SLA breach risk, include "reason": "sla_breach"
    - If unsure, type: hint
    - Read thread_history for context on follow-up emails
    """
).strip()


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------


def build_user_prompt(
    task_description: str,
    email_subject: str,
    email_sender: str,
    email_body: str,
    thread_history: list,
    step: int,
    last_result: str,
    last_reward: float,
    history: List[str],
) -> str:
    thread_block = ""
    if thread_history:
        parts = []
        for msg in thread_history:
            if hasattr(msg, "sender"):
                parts.append(
                    f"  [{msg.timestamp}] From: {msg.sender}\n"
                    f"  Subject: {msg.subject}\n"
                    f"  {msg.body}"
                )
            elif isinstance(msg, dict):
                parts.append(
                    f"  [{msg.get('timestamp', '')}] From: {msg.get('sender', '')}\n"
                    f"  Subject: {msg.get('subject', '')}\n"
                    f"  {msg.get('body', '')}"
                )
        if parts:
            thread_block = "\n\nThread history (oldest first):\n" + "\n---\n".join(parts)

    history_block = "\n".join(history[-6:]) if history else "None"
    return textwrap.dedent(
        f"""
        TASK: {task_description}

        Current Email:
          From: {email_sender}
          Subject: {email_subject}
          Body: {email_body}
        {thread_block}

        Step: {step}
        Last result: {last_result!r}
        Last reward: {last_reward:.3f}

        Previous actions this episode:
        {history_block}

        Send your next action JSON.
        """
    ).strip()


# ---------------------------------------------------------------------------
# Model call
# ---------------------------------------------------------------------------


def get_model_action(
    task_description: str,
    email_subject: str,
    email_sender: str,
    email_body: str,
    thread_history: list,
    step: int,
    last_result: str,
    last_reward: float,
    history: List[str],
) -> str:
    user_prompt = build_user_prompt(
        task_description,
        email_subject,
        email_sender,
        email_body,
        thread_history,
        step,
        last_result,
        last_reward,
        history,
    )
    try:
        completion = client_llm.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.2,
        )
        text = (completion.choices[0].message.content or "").strip()
        # Strip markdown code fences if the model wraps the command
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(ln for ln in lines if not ln.startswith("```")).strip()
        if text.startswith("{"):
            return text
        # Fallback safe default
        return '{"action": "classify", "priority": "medium", "category": "general"}'
    except Exception as exc:
        print(f"[DEBUG] Model call failed: {exc}", flush=True)
        return '{"action": "classify", "priority": "medium", "category": "general"}'


# ---------------------------------------------------------------------------
# Episode runner — mirrors reference run_task() exactly
# ---------------------------------------------------------------------------


def run_task(env_url: str) -> None:
    with EmailTriageEnv(base_url=env_url).sync() as env:
        for _ in range(NUM_EPISODES):
            result = env.reset()
            obs: EmailTriageObservation = result.observation

            task_id = str(obs.task.task_id) if obs.task else "unknown"
            task_description = obs.task.description if obs.task else ""
            email_subject = obs.task.email_subject if obs.task else ""
            email_sender = obs.task.email_sender if obs.task else ""
            email_body = obs.task.email_body if obs.task else ""
            thread_history = list(obs.task.thread_history) if obs.task else []

            last_result = obs.last_action_result
            last_reward = 0.0
            history: List[str] = []
            rewards: List[float] = []
            steps = 0
            success = False

            print(f"[START] task={task_id} env={BENCHMARK} model={MODEL_NAME}", flush=True)

            for step in range(1, MAX_STEPS + 1):
                if result.done:
                    break

                action = get_model_action(
                    task_description,
                    email_subject,
                    email_sender,
                    email_body,
                    thread_history,
                    step,
                    last_result,
                    last_reward,
                    history,
                )

                result = env.step(EmailTriageAction(action=action))
                obs = result.observation

                reward = obs.reward or 0.0
                done = result.done
                last_result = obs.last_action_result
                last_reward = reward

                # Clamp reward to strictly (0, 1) — never exactly 0.0 or 1.0
                if reward <= 0.0:
                    reward = 0.001
                elif reward >= 1.0:
                    reward = 0.999

                rewards.append(reward)
                steps = step

                done_str = "true" if done else "false"
                error_str = None if obs.last_action_valid else obs.last_action_result
                error_val = error_str if error_str else "null"
                print(
                    f"[STEP] step={step} action={action!r} reward={reward:.2f} "
                    f"done={done_str} error={error_val}",
                    flush=True,
                )

                history.append(
                    f"Step {step}: {action} -> reward={reward:.3f} | {last_result}"
                )

                if obs.task_achieved:
                    success = True
                    break

                if done:
                    break

            score = max(rewards) if rewards else 0.001
            # Clamp score to strictly (0, 1)
            score = min(max(score, 0.001), 0.999)

            success_str = "true" if success else "false"
            rewards_str = ",".join(f"{r:.2f}" for r in rewards)
            print(
                f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}",
                flush=True,
            )


if __name__ == "__main__":
    run_task(ENV_URL)
