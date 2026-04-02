# ============================================================
# Dockerfile — Advanced Enterprise Email Triage OpenEnv
# Compatible with Hugging Face Spaces Docker SDK
# Exposes port 7860
#
# SECURITY: No API keys, tokens, or secrets are copied into
# this image. All credentials must be provided at runtime via
# environment variables (HF Space Secrets or docker -e flags).
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for layer caching
COPY requirements.txt .

# Install Python dependencies into a prefix
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Non-root user for HF Spaces security best practice
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
# NOTE: .dockerignore prevents .env, secrets, and __pycache__ from being copied
COPY models.py .
COPY data.py .
COPY tools.py .
COPY graders.py .
COPY rewards.py .
COPY environment.py .
COPY inference.py .
COPY openenv.yaml .
COPY server/ ./server/

# Set ownership
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Expose port for HF Spaces
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/health').raise_for_status()"

# Environment defaults (non-secret — safe to bake in)
ENV HOST=0.0.0.0
ENV PORT=7860
ENV LOG_LEVEL=info
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Secrets must be provided at runtime via HF Space Secrets or docker -e ──
# ENV OPENAI_API_KEY is NOT set here
# ENV API_BASE_URL is NOT set here
# ENV MODEL_NAME is NOT set here

# Start the FastAPI server
CMD ["python", "-m", "uvicorn", "server.app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--log-level", "info", \
     "--no-access-log"]
