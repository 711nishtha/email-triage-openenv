#!/bin/bash
# start.sh - start both backend and UI concurrently

# Start FastAPI backend in the background on port 8000
echo "Starting FastAPI backend on port 8000..."
uvicorn server.app:app --host 0.0.0.0 --port 8000 --no-access-log &
BACKEND_PID=$!

# Wait for backend to be healthy
until curl -s http://localhost:8000/health > /dev/null; do
  echo "Waiting for backend..."
  sleep 2
done
echo "Backend is ready!"

# Start Streamlit UI on the exposed port (7860)
echo "Starting Streamlit UI on port 7860..."
export BACKEND_URL="http://localhost:8000"
streamlit run ui_streamlit/app.py --server.port 7860 --server.address 0.0.0.0

# Cleanup on exit
trap "kill $BACKEND_PID" EXIT
