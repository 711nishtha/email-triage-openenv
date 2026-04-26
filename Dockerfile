# ================================================================
# Dockerfile — Email Triage RL Environment (Grand Finale)
# Optimised for HuggingFace Spaces (port 7860)
# Two-stage build: fast uv install → slim runtime image
# ================================================================

# Stage 1: dependency installer (uv for speed + reliability)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy dependency manifests first for best layer caching
COPY requirements.txt pyproject.toml ./

# Install all Python dependencies into system Python
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------
# Stage 2: slim runtime image
# ----------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /usr/local/lib/python3.12/site-packages \
                    /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy application source
COPY . .

# ---- Environment defaults (all overridable at runtime) ----
# API_BASE_URL and API_KEY are injected by the hackathon proxy at runtime.
# Do NOT hardcode real API keys here.
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=false
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
ENV SOCIETY_MODE=full
ENV MAX_STEPS=10
ENV NUM_EPISODES=11

# HuggingFace Spaces requires port 7860
EXPOSE 7860

# Health check — validator uses /reset not /health, but good practice
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')" \
  || exit 1

# Start FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", \
     "--log-level", "warning", "--no-access-log"]
