FROM python:3.11-slim AS base

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose API + Streamlit ports
EXPOSE 8001 8501

# Default: run the API server
CMD ["uvicorn", "api.run_api:app", "--host", "0.0.0.0", "--port", "8001"]


# ---------------------------------------------------------------------------
# GPU variant (for training)
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS gpu

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN python3.11 -m pip install --no-cache-dir --upgrade pip && \
    python3.11 -m pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8001 8501

CMD ["python3.11", "-m", "uvicorn", "api.run_api:app", "--host", "0.0.0.0", "--port", "8001"]
