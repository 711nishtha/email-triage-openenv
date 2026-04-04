FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Make start script executable
RUN chmod +x start.sh

# Expose ports
# 7860 — FastAPI backend (primary)
# 8501 — Streamlit UI (optional)
EXPOSE 7860
EXPOSE 8501

# Health check on backend
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Entry point
CMD ["./start.sh"]
