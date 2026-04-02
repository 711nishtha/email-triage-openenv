"""
inference.py — Agent runner for Advanced Enterprise Email Triage OpenEnv.

Uses the OpenAI client to run an LLM agent against all three tasks.
Strictly follows the hackathon logging format:
  [START], [STEP], [END] with structured fields.

Credentials are loaded ONLY from environment variables.
The script fails gracefully with a clear message if required vars are missing.

Usage:
  python inference.py                    # run all tasks
  python inference.py --task easy
  python inference.py --task medium
  python inference.py --task hard
  python inference.py --base-url http://localhost:7860  # run against local server
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# ── Credential loading ───────────────────────────────────────────────────────

def load_credentials() -> Dict[str, str]:
    """
    Load all required credentials from environment variables.
    Exits with a clear error message if any required var is missing.
    Never logs actual secret values.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    api_base = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    # HF_TOKEN is optional — only needed for private model access
    hf_token = os.getenv("HF_TOKEN")

    if not api_key:
        print(
            "\n[ERROR] Required environment variable OPENAI_API_KEY is not set.\n"
            "Please set it before running inference.py:\n"
            "  export OPENAI_API_KEY=your_key_here\n"
            "Or copy .env.example to .env and fill in your values.\n"
            "NEVER hardcode API keys in source files.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[INFO] Credentials loaded — OPENAI_API_KEY: ***{api_key[-4:]}")
    print(f"[INFO] API_BASE_URL: {api_base}")
    print(f"[INFO] MODEL_NAME: {model_name}")
    if hf_token:
        print(f"[INFO] HF_TOKEN: ***{hf_token[-4:]}")

    return {
        "api_key": api_key,
        "api_base": api_base,
        "model_name": model_name,
        "hf_token": hf_token,
    }


# ── Hackathon logging format ─────────────────────────────────────────────────

def log_start(task: str, env: str, model: str, extra: Optional[Dict] = None) -> None:
    """
    [START] log — called once at the beginning of an episode.

    Format: [START] task=<task> env=<env> model=<model> timestamp=<iso> [extra...]
    """
    ts = datetime.utcnow().isoformat() + "Z"
    fields = f"task={task} env={env} model={model} timestamp={ts}"
    if extra:
        for k, v in extra.items():
            fields += f" {k}={v}"
    print(f"[START] {fields}", flush=True)


def log_step(
    step: int,
    action_type: str,
    reward: float,
    cumulative_score: float,
    done: bool,
    extra: Optional[Dict] = None,
) -> None:
    """
    [STEP] log — called after each environment step.

    Format: [STEP] step=<n> action=<type> reward=<r> score=<s> done=<bool> [extra...]
    """
    fields = (
        f"step={step} action={action_type} reward={reward:.4f} "
        f"score={cumulative_score:.4f} done={done}"
    )
    if extra:
        for k, v in extra.items():
            fields += f" {k}={json.dumps(v) if isinstance(v, (dict, list)) else v}"
    print(f"[STEP] {fields}", flush=True)


def log_end(
    task: str,
    env: str,
    model: str,
    rewards: List[float],
    score: float,
    success: bool,
    total_steps: int,
    elapsed_seconds: float,
    extra: Optional[Dict] = None,
) -> None:
    """
    [END] log — called once at episode conclusion.

    Format: [END] task=<task> env=<env> model=<model> score=<s> success=<bool>
                  steps=<n> elapsed=<sec> rewards=[...] [extra...]
    """
    ts = datetime.utcnow().isoformat() + "Z"
    rewards_str = json.dumps([round(r, 4) for r in rewards])
    fields = (
        f"task={task} env={env} model={model} "
        f"score={score:.4f} success={success} "
        f"steps={total_steps} elapsed={elapsed_seconds:.2f}s "
        f"rewards={rewards_str} timestamp={ts}"
    )
    if extra:
        for k, v in extra.items():
            fields += f" {k}={json.dumps(v) if isinstance(v, (dict, list)) else v}"
    print(f"[END] {fields}", flush=True)


# ── Prompt builder ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert enterprise email triage agent. Your job is to:
1. Analyse each email in the inbox carefully
2. Assign correct priority: critical, high, medium, low, or spam
3. Assign correct category: security_incident, executive_request, hr_matter, vendor_contract,
   it_support, team_update, customer_escalation, phishing, newsletter, or other
4. Route to the correct team: security_team, ceo_office, cfo_office, hr_team, legal_team,
   it_helpdesk, engineering_team, sales_team, finance_team, operations_team,
   customer_success, executive_assistant, archive, or spam_folder
5. Use available tools when needed (calendar_check, kb_search, sender_lookup)
6. Escalate genuine security threats immediately

CRITICAL SAFETY RULES:
- Any email requesting credentials/passwords from an external domain is PHISHING
- Wire transfer requests via email require verification — flag as phishing if unsolicited
- Spoofed domains (acmecorp-security.com, acme-corp.net) are NOT legitimate internal domains
- BEC (Business Email Compromise): board/exec asking for secret wire transfers = phishing
- Real internal IT will NEVER ask for passwords via email

OUTPUT FORMAT: Respond with a single valid JSON object matching the TriageAction schema.
Do NOT include markdown fences. Do NOT add commentary outside the JSON.

Example triage:
{"action_type": "triage", "email_id": "abc12345", "priority": "high",
 "category": "vendor_contract", "route_to": "legal_team",
 "reasoning": "External vendor requesting DPA review for GDPR compliance."}

Example tool use:
{"action_type": "use_tool", "tool_name": "sender_lookup",
 "tool_params": {"email": "suspicious@acmecorp-security.com"}}

Example escalation:
{"action_type": "escalate", "email_id": "xyz98765",
 "escalation_target": "security_team",
 "escalation_reason": "Suspected phishing: external domain spoofing CEO identity requesting wire transfer"}

When done with all emails:
{"action_type": "done"}
"""


def build_user_prompt(observation: Dict[str, Any]) -> str:
    """Convert the environment observation into a clear prompt for the LLM."""
    inbox = observation.get("inbox", [])
    thread_history = observation.get("thread_history", {})
    sender_profiles = observation.get("sender_profiles", {})
    tools = observation.get("available_tools", [])
    step = observation.get("step", 0)
    max_steps = observation.get("max_steps", 10)
    score = observation.get("current_score", 0.0)
    warnings = observation.get("warnings", [])

    lines = [
        f"=== EMAIL TRIAGE INBOX ===",
        f"Step: {step}/{max_steps} | Current score: {score:.3f}",
        f"Emails pending: {len(inbox)}",
        f"Available tools: {tools or 'none'}",
    ]

    if warnings:
        lines.append(f"\n⚠️ WARNINGS: {'; '.join(warnings)}")

    # Thread history
    if thread_history:
        lines.append("\n--- THREAD HISTORY (context only) ---")
        for thread_id, messages in thread_history.items():
            lines.append(f"Thread {thread_id}:")
            for msg in messages:
                lines.append(
                    f"  [{msg['timestamp'][:10]}] {msg['sender_display_name']} "
                    f"→ {', '.join(msg['recipients'][:2])}: {msg['subject']}"
                )
                lines.append(f"    {msg['body'][:150]}...")

    # Inbox emails
    lines.append("\n--- INBOX EMAILS TO TRIAGE ---")
    for email in inbox:
        profile = sender_profiles.get(email["sender"], {})
        rep_score = profile.get("reputation_score", 0.5)
        is_vip = profile.get("is_vip", False)
        is_internal = profile.get("is_internal", False)
        is_flagged = profile.get("is_flagged_suspicious", False)

        sender_flags = []
        if is_vip:
            sender_flags.append("VIP")
        if is_internal:
            sender_flags.append("INTERNAL")
        if is_flagged:
            sender_flags.append("⚠️SUSPICIOUS")
        if rep_score < 0.3:
            sender_flags.append(f"LOW_REP:{rep_score:.2f}")

        lines.append(f"\n[EMAIL ID: {email['email_id']}]")
        lines.append(f"Subject: {email['subject']}")
        lines.append(
            f"From: {email['sender_display_name']} <{email['sender']}> "
            f"[{', '.join(sender_flags) or 'unknown'}]"
        )
        lines.append(f"To: {', '.join(email['recipients'][:3])}")
        lines.append(f"Time: {email['timestamp']}")
        if email.get("thread_id"):
            lines.append(f"Thread: {email['thread_id']}")
        lines.append(f"Body:\n{email['body']}")
        if email.get("links"):
            lines.append(f"Links: {email['links']}")
        if email.get("has_attachments"):
            lines.append(f"Attachments: {email.get('attachment_names', [])}")

    lines.append("\n--- END OF INBOX ---")
    lines.append("\nRespond with a single JSON action (triage, use_tool, escalate, or done).")
    return "\n".join(lines)


# ── Environment HTTP client ──────────────────────────────────────────────────

class EnvClient:
    """HTTP client for the FastAPI environment server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=30.0)

    def reset(self, task_id: str, seed: Optional[int] = None) -> Dict[str, Any]:
        payload = {"task_id": task_id}
        if seed is not None:
            payload["seed"] = seed
        r = self.client.post(f"{self.base_url}/reset", json=payload)
        r.raise_for_status()
        return r.json()

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        r = self.client.post(f"{self.base_url}/step", json=action)
        r.raise_for_status()
        return r.json()

    def state(self) -> Dict[str, Any]:
        r = self.client.get(f"{self.base_url}/state")
        r.raise_for_status()
        return r.json()

    def close(self):
        self.client.close()


# ── Agent loop ───────────────────────────────────────────────────────────────

def run_task(
    task_id: str,
    openai_client: OpenAI,
    env_client: EnvClient,
    model_name: str,
    env_name: str = "advanced-enterprise-email-triage",
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """
    Run a complete episode for one task.

    Returns summary dict with score, success, steps, rewards.
    """
    log_start(
        task=task_id,
        env=env_name,
        model=model_name,
        extra={"seed": seed},
    )

    start_time = time.time()
    rewards: List[float] = []
    step_count = 0

    # ── Reset environment ────────────────────────────────────────────────────
    try:
        obs = env_client.reset(task_id=task_id, seed=seed)
    except Exception as e:
        print(f"[ERROR] Failed to reset environment: {e}", file=sys.stderr)
        raise

    conversation_history = []  # Full LLM conversation for multi-turn context
    done = obs.get("done", False)
    max_steps = obs.get("max_steps", 25)
    final_score = 0.0

    # ── Episode loop ─────────────────────────────────────────────────────────
    while not done and step_count < max_steps:
        # Build prompt from observation
        user_prompt = build_user_prompt(obs)
        conversation_history.append({"role": "user", "content": user_prompt})

        # Call LLM
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *conversation_history,
                ],
                temperature=0.2,
                max_tokens=512,
            )
            raw_content = response.choices[0].message.content or ""
        except Exception as e:
            print(f"[ERROR] LLM call failed at step {step_count}: {e}", file=sys.stderr)
            break

        # Parse action JSON
        try:
            # Strip any accidental markdown fences
            cleaned = raw_content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]
            action_dict = json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"[WARN] Step {step_count}: JSON parse failed: {e}. Raw: {raw_content[:200]}")
            # Fallback: emit done action to end episode cleanly
            action_dict = {"action_type": "done"}

        # Add assistant turn to conversation history
        conversation_history.append({"role": "assistant", "content": raw_content})

        # Submit action to environment
        try:
            step_result = env_client.step(action_dict)
        except Exception as e:
            print(f"[ERROR] Step {step_count} failed: {e}", file=sys.stderr)
            break

        obs = step_result.get("observation", {})
        reward = step_result.get("reward", 0.0)
        done = step_result.get("done", False)
        info = step_result.get("info", {})

        rewards.append(reward)
        step_count += 1
        final_score = obs.get("current_score", 0.0)

        # Collect signals for logging
        signals = info.get("signals", [])
        signals_preview = signals[:3] if signals else []

        log_step(
            step=step_count,
            action_type=action_dict.get("action_type", "unknown"),
            reward=reward,
            cumulative_score=final_score,
            done=done,
            extra={
                "email_id": action_dict.get("email_id", ""),
                "signals": signals_preview,
            },
        )

        # If the environment returned a tool result, inject it into the conversation
        if action_dict.get("action_type") == "use_tool" and info.get("signals"):
            tool_feedback = "\n".join(info["signals"])
            conversation_history.append({
                "role": "user",
                "content": f"[TOOL RESULT]\n{tool_feedback}\n\nContinue triaging."
            })

    # ── Episode end ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    # Fetch final grader breakdown from state
    grader_breakdown: Dict[str, Any] = {}
    try:
        state = env_client.state()
        grader_breakdown = state.get("grader_breakdown", {})
        final_score = state.get("total_reward", final_score)
    except Exception:
        pass

    # Determine success threshold per task
    success_thresholds = {"easy": 0.60, "medium": 0.50, "hard": 0.40}
    success = final_score >= success_thresholds.get(task_id, 0.50)

    log_end(
        task=task_id,
        env=env_name,
        model=model_name,
        rewards=rewards,
        score=final_score,
        success=success,
        total_steps=step_count,
        elapsed_seconds=elapsed,
        extra={"grader_breakdown": grader_breakdown},
    )

    return {
        "task_id": task_id,
        "score": final_score,
        "success": success,
        "steps": step_count,
        "rewards": rewards,
        "elapsed": elapsed,
        "grader_breakdown": grader_breakdown,
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run the Advanced Enterprise Email Triage agent."
    )
    parser.add_argument(
        "--task",
        choices=["easy", "medium", "hard", "all"],
        default="all",
        help="Task to run (default: all)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:7860",
        help="Environment server base URL (default: http://localhost:7860)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible episodes (default: 42)",
    )
    args = parser.parse_args()

    # Load credentials
    creds = load_credentials()

    # Build OpenAI client
    openai_client = OpenAI(
        api_key=creds["api_key"],
        base_url=creds["api_base"],
    )

    # Build environment client
    env_client = EnvClient(base_url=args.base_url)

    # Determine tasks to run
    tasks = ["easy", "medium", "hard"] if args.task == "all" else [args.task]

    all_results = []
    for task_id in tasks:
        print(f"\n{'='*60}", flush=True)
        print(f"Running task: {task_id.upper()}", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            result = run_task(
                task_id=task_id,
                openai_client=openai_client,
                env_client=env_client,
                model_name=creds["model_name"],
                seed=args.seed,
            )
            all_results.append(result)
        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user.", flush=True)
            break
        except Exception as e:
            print(f"[ERROR] Task {task_id} failed: {e}", file=sys.stderr)
            traceback.print_exc()
            all_results.append({
                "task_id": task_id, "score": 0.0, "success": False,
                "error": str(e)
            })

    # Print summary
    print(f"\n{'='*60}", flush=True)
    print("INFERENCE SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    for r in all_results:
        status = "✓ PASS" if r.get("success") else "✗ FAIL"
        print(
            f"  {status} [{r['task_id'].upper():6s}] "
            f"score={r.get('score', 0.0):.4f} "
            f"steps={r.get('steps', 0)} "
            f"elapsed={r.get('elapsed', 0.0):.1f}s",
            flush=True,
        )

    env_client.close()

    # Exit with code 1 if any task failed
    if any(not r.get("success") for r in all_results):
        sys.exit(1)


if __name__ == "__main__":
    main()
