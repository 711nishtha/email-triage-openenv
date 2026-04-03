# ============================================================
# Dockerfile — Advanced Enterprise Email Triage OpenEnv
# Optimized for HF Spaces and mandatory validator compliance
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /install /usr/local

COPY models.py .
COPY data.py .
COPY tools.py .
COPY graders.py .
COPY rewards.py .
COPY environment.py .
COPY openenv.yaml .
COPY start.sh .
COPY inference.py .
COPY ui_streamlit/ ./ui_streamlit/
COPY server/ ./server/

RUN chown -R appuser:appuser /app
RUN chmod +x start.sh

USER appuser

# Standard OpenEnv port
EXPOSE 7860

# Health check hits the FastAPI health endpoint directly on 7860
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:7860/health || exit 1

ENV HOST=0.0.0.0
ENV PORT=7860
ENV BACKEND_URL=http://localhost:7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["./start.sh"]
