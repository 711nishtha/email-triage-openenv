#!/bin/bash
set -e

echo "==================================================="
echo "  Advanced Enterprise Email Triage OpenEnv"
echo "  Backend: http://0.0.0.0:7860"
echo "  UI:      http://0.0.0.0:8501 (optional)"
echo "==================================================="

# ── Start FastAPI backend ────────────────────────────
echo "[INFO] Starting FastAPI backend on port 7860..."
cd /app
python -m uvicorn server.app:app \
    --host 0.0.0.0 \
    --port 7860 \
    --workers 1 \
    --log-level info &

BACKEND_PID=$!
echo "[INFO] Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "[INFO] Waiting for backend to be ready..."
for i in $(seq 1 30); do
    if curl -s http://localhost:7860/health > /dev/null 2>&1; then
        echo "[INFO] Backend is ready!"
        break
    fi
    echo "[INFO] Attempt $i/30 — backend not ready yet, waiting..."
    sleep 1
done

# ── Start Streamlit UI (optional) ────────────────────
echo "[INFO] Starting Streamlit UI on port 8501..."
streamlit run /app/ui.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --browser.gatherUsageStats false &

UI_PID=$!
echo "[INFO] Streamlit UI PID: $UI_PID"

# ── Keep container alive ──────────────────────────────
echo "[INFO] Both services started. Container running."
echo "[INFO] Backend: http://0.0.0.0:7860"
echo "[INFO] UI:      http://0.0.0.0:8501"

# Wait for either process to exit — if backend dies, restart it
# If UI dies, that's OK — backend continues
wait -n $BACKEND_PID $UI_PID

# If backend died, exit with error so container restarts
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "[ERROR] Backend process died. Exiting."
    exit 1
fi

# UI died but backend is fine — just wait on backend
echo "[WARNING] UI process ended. Backend still running."
wait $BACKEND_PID
