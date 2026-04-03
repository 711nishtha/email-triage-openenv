#!/bin/bash

# run_all.sh — Start both FastAPI backend and Streamlit UI locally

# 1. Kill any existing processes on ports 7860 and 8501
fuser -k 7860/tcp 2>/dev/null
fuser -k 8501/tcp 2>/dev/null

# 2. Start FastAPI backend in the background
echo "Starting FastAPI backend on http://localhost:7860..."
export PORT=7860
export HOST=0.0.0.0
python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 > backend.log 2>&1 &
BACKEND_PID=$!

# 3. Wait for backend to be healthy
echo "Waiting for backend to be ready..."
until curl -s http://localhost:7860/health > /dev/null; do
  sleep 1
done
echo "Backend is ready!"

# 4. Start Streamlit UI
echo "Starting Streamlit UI on http://localhost:8501..."
export BACKEND_URL="http://localhost:7860"
streamlit run ui_streamlit/app.py --server.port 8501

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT
