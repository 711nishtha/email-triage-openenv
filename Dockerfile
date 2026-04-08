ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

ARG BUILD_MODE=standalone
ARG ENV_NAME=email_triage

COPY . /app/env

WORKDIR /app/env

# Ensure uv is available
RUN if ! command -v uv >/dev/null 2>&1; then \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/uv && \
    mv /root/.local/bin/uvx /usr/local/bin/uvx; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra dev --no-install-project --no-editable || \
    pip install --no-cache-dir -r requirements.txt

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --extra dev --no-editable || true

# ---- Final runtime stage ----
FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /root/.local/share/uv/python /root/.local/share/uv/python 2>/dev/null || true
COPY --from=builder /app/env/.venv /app/.venv 2>/dev/null || true
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"
ENV PYTHONUNBUFFERED=1
ENV ENABLE_WEB_INTERFACE=false

# HuggingFace Spaces requires port 7860
EXPOSE 7860

ENV API_BASE_URL=https://router.huggingface.co/v1
ENV MODEL_NAME=gpt-4o-mini

CMD ["sh", "-c", "uvicorn server.app:app --host 0.0.0.0 --port 7860"]
