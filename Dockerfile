# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Install package in editable mode
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install -e .

# Copy the rest (scripts, dashboard, examples)
COPY scripts ./scripts
COPY dashboard ./dashboard
COPY examples ./examples

# Default command runs a small end-to-end pipeline
CMD ["python", "scripts/run_end_to_end.py", "--n_borrowers", "2000", "--months", "3"]
