#!/bin/bash
# start.sh - start both backend and UI, ensuring backend is on 7860 for validator

# 1. Start FastAPI backend on port 7860 (Mandatory for OpenEnv validator)
echo "Starting FastAPI backend on port 7860..."
uvicorn server.app:app --host 0.0.0.0 --port 7860 --no-access-log &
BACKEND_PID=$!

# 2. Start Streamlit UI on port 8501
# Note: On HF Spaces, only 7860 is externally visible by default. 
# The UI is kept running internally as requested.
echo "Starting Streamlit UI on port 8501..."
export BACKEND_URL="http://localhost:7860"
streamlit run ui_streamlit/app.py --server.port 8501 --server.address 0.0.0.0 &
UI_PID=$!

# 3. Wait for backend to be healthy
echo "Waiting for backend on 7860..."
until curl -s http://localhost:7860/health > /dev/null; do
  sleep 2
done
echo "Backend is ready at http://localhost:7860"

# Cleanup on exit
trap "kill $BACKEND_PID $UI_PID" EXIT

# Keep script alive
wait $BACKEND_PID
