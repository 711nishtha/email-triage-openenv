"""
inference.py — Mandatory agent runner for the OpenEnv Hackathon.
Matches the EXACT stdout format and environment variable requirements.
"""

import asyncio
import os
import json
import sys
import textwrap
from typing import List, Optional, Dict, Any

import httpx
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# ── Mandatory Environment Variables ──────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# Task Configuration
TASKS = ["easy", "medium", "hard"]
MAX_STEPS = 25
TEMPERATURE = 0.2

# ── Mandatory Logging Format ─────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    """[START] task=<task_name> env=<benchmark> model=<model_name>"""
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>"""
    error_val = error if error else "null"
    done_val = str(done).lower()
    # Ensure action string is compact and has no newlines
    action_clean = action.replace("\n", " ").strip()[:100]
    print(
        f"[STEP] step={step} action={action_clean} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    """[END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    success_val = str(success).lower()
    print(f"[END] success={success_val} steps={steps} rewards={rewards_str}", flush=True)

# ── Agent Logic ──────────────────────────────────────────────────────────────

def build_minimal_prompt(obs: Dict[str, Any]) -> str:
    """Simplified prompt for the triage agent."""
    inbox = obs.get("inbox", [])
    tools = obs.get("available_tools", [])
    
    prompt = f"Inbox has {len(inbox)} emails.\n"
    for e in inbox[:3]:  # Limit context for speed
        prompt += f"- ID: {e['email_id']} | From: {e['sender']} | Sub: {e['subject']}\n"
    
    prompt += f"\nAvailable tools: {tools}\n"
    prompt += "Respond with ONE JSON action (triage, use_tool, escalate, or done)."
    return prompt

async def run_episode(task_id: str, client: OpenAI):
    """Run a single episode against the environment server."""
    async with httpx.AsyncClient(timeout=60.0) as http:
        # 1. Reset
        try:
            resp = await http.post(f"{ENV_URL}/reset", json={"task_id": task_id})
            resp.raise_for_status()
            obs = resp.json()
        except Exception as e:
            print(f"[DEBUG] Reset failed: {e}", file=sys.stderr)
            log_end(False, 0, [])
            return

        log_start(task=task_id, env="advanced-enterprise-email-triage", model=MODEL_NAME)
        
        rewards = []
        done = False
        step_count = 0
        
        while not done and step_count < MAX_STEPS:
            step_count += 1
            user_prompt = build_minimal_prompt(obs)
            
            # 2. Get Model Action
            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are an email triage agent. Output JSON only."},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=TEMPERATURE,
                    max_tokens=300,
                )
                raw_action = (completion.choices[0].message.content or "").strip()
                
                # Cleanup JSON
                if "```" in raw_action:
                    raw_action = raw_action.split("```")[1]
                    if raw_action.startswith("json"):
                        raw_action = raw_action[4:]
                    raw_action = raw_action.split("```")[0].strip()
                
                action_dict = json.loads(raw_action)
            except Exception as e:
                action_dict = {"action_type": "done"}
                raw_action = "done"

            # 3. Step Environment
            try:
                step_resp = await http.post(f"{ENV_URL}/step", json=action_dict)
                step_resp.raise_for_status()
                data = step_resp.json()
                
                obs = data.get("observation", {})
                reward = data.get("reward", 0.0)
                done = data.get("done", False)
                error = None
            except Exception as e:
                reward = 0.0
                done = True
                error = str(e)

            rewards.append(reward)
            log_step(step=step_count, action=raw_action, reward=reward, done=done, error=error)
            
            if done:
                break

        # 4. Success check from state
        success = False
        try:
            state_resp = await http.get(f"{ENV_URL}/state")
            if state_resp.status_code == 200:
                state = state_resp.json()
                # Use normalized total reward for success check
                success = state.get("total_reward", 0.0) >= 0.4
        except:
            pass
        
        log_end(success=success, steps=step_count, rewards=rewards)

async def main():
    if not HF_TOKEN:
        print("[ERROR] HF_TOKEN environment variable is required.")
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    
    # Run all tasks sequentially
    for task in TASKS:
        await run_episode(task, client)

if __name__ == "__main__":
    asyncio.run(main())
