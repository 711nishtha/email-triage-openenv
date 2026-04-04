"""
server/app.py — FastAPI application for the Email Triage OpenEnv.

Endpoints:
  POST /reset  → Observation
  POST /step   → StepResponse
  GET  /state  → EnvironmentState
  GET  /health → {"status": "ok"}
"""

from __future__ import annotations
import sys
import os

# Add parent to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from models import (
    Action,
    Observation,
    StepResponse,
    EnvironmentState,
    ResetRequest,
)
from environment import EmailTriageEnvironment


# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="Advanced Enterprise Email Triage OpenEnv",
    description="OpenEnv-compliant environment for evaluating email triage agents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance (single-session server)
env = EmailTriageEnvironment()


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "service": "email-triage-openenv", "version": "1.0.0"}


@app.post("/reset", response_model=Observation)
async def reset(request: ResetRequest = None) -> Observation:
    """
    Reset the environment for a new episode.
    
    Args:
        task_id: One of "easy", "medium", "hard" (default: "easy")
    
    Returns:
        Initial observation with inbox emails and available tools.
    """
    task_id = "easy"
    if request and request.task_id:
        task_id = request.task_id

    valid_tasks = ["easy", "medium", "hard"]
    if task_id not in valid_tasks:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id '{task_id}'. Must be one of: {valid_tasks}"
        )

    observation = env.reset(task_id=task_id)
    return observation


@app.post("/step", response_model=StepResponse)
async def step(action: Action) -> StepResponse:
    """
    Submit a triage action and receive a reward.
    
    Action types:
      - triage: Classify an email with priority, category, and route
      - escalate: Escalate an email directly to a team
      - use_tool: Invoke an investigation tool
      - done: Signal that triage is complete
    
    Returns:
        observation, reward (0.0–1.0), done flag, and info dict.
    """
    if env.task_id == "" or (env.step_count == 0 and not env.emails):
        raise HTTPException(
            status_code=400,
            detail="Environment not initialized. Call /reset first."
        )

    try:
        observation, reward, done, info = env.step(action)
    except Exception as e:
        obs_fallback = env._build_observation(message=f"Internal error: {str(e)[:120]}")
        return StepResponse(
            observation=obs_fallback,
            reward=0.0,
            done=False,
            info={"error": str(e)[:200]},
        )
    return StepResponse(
        observation=observation,
        reward=reward,
        done=done,
        info=info,
    )


@app.get("/state", response_model=EnvironmentState)
async def state() -> EnvironmentState:
    """
    Get the full current environment state.
    
    Returns complete state including inbox, triaged emails,
    cumulative reward, step count, and tool results.
    """
    return env.state()


# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )
