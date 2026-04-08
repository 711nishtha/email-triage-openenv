# ====================== Dockerfile for HF Spaces ======================
# Optimized for Hugging Face Spaces + OpenEnv email-triage project
# Uses uv for fast & reliable dependency installation

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy dependency files first (best caching)
COPY pyproject.toml uv.lock requirements.txt ./

# Install dependencies using uv (with cache mount for speed & reliability)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system --no-cache-dir -r requirements.txt

# ---- Final runtime stage ----
FROM python:3.12-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin/uvicorn /usr/local/bin/uvicorn

# Copy the application code
COPY . .

# Environment variables for HF Spaces
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=false
ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=gpt-4o-mini

# Hugging Face Spaces requires port 7860
EXPOSE 7860

# Start the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
