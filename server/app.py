"""
server/app.py — FastAPI server for the Advanced Enterprise Email Triage OpenEnv.

Exposes the standard OpenEnv HTTP interface:
  POST /reset   → Reset environment, return initial observation
  POST /step    → Submit action, receive observation + reward
  GET  /state   → Return full current environment state
  GET  /health  → Liveness probe

Designed for HF Spaces Docker deployment on port 7860.
No secrets are logged or stored.
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is on the path (needed for imports when running from server/)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import (
    EmailObservation,
    EnvironmentState,
    ResetRequest,
    StepResponse,
    TriageAction,
)
from environment import EmailTriageEnvironment


# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Advanced Enterprise Email Triage — OpenEnv",
    description=(
        "A production-grade enterprise email triage environment for AI agents. "
        "Supports three difficulty levels: easy, medium, hard."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow all origins for HF Spaces / public API access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Environment instance (single-session for hackathon) ──────────────────────
# For production multi-session, use a session store keyed by session_id.
env = EmailTriageEnvironment()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> Dict[str, str]:
    """Liveness probe — always returns 200 OK."""
    return {"status": "ok", "service": "advanced-enterprise-email-triage-openenv"}


@app.post("/reset", response_model=EmailObservation)
async def reset(request: ResetRequest) -> EmailObservation:
    """
    Reset the environment to a fresh episode.

    Body:
      task_id: "easy" | "medium" | "hard"
      seed: optional int for reproducible episodes

    Returns the initial observation.
    """
    try:
        observation = env.reset(request)
        return observation
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Reset failed: {exc}") from exc


@app.post("/step", response_model=StepResponse)
async def step(action: TriageAction) -> StepResponse:
    """
    Submit one agent action and receive the next observation.

    Returns: observation, reward (float), done (bool), info (dict).
    """
    try:
        response = env.step(action)
        return response
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Step failed: {exc}") from exc


@app.get("/state", response_model=EnvironmentState)
async def state() -> EnvironmentState:
    """
    Return the full current environment state.

    Includes ground-truth labels for post-episode evaluation.
    Not intended for the agent to call during inference.
    """
    try:
        return env.state()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"State retrieval failed: {exc}") from exc


# ── Error handlers ───────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error: {str(exc)}", "type": type(exc).__name__},
    )


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    # Load optional config from environment (no secrets needed for server itself)
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "7860"))
    log_level = os.getenv("LOG_LEVEL", "info")

    uvicorn.run(
        "server.app:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=False,
    )
