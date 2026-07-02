# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# AI Candidate-Ranking System — reproducible container
#
#   docker compose up --build
#
# Build-time (network allowed): installs deps and BAKES the embedding model
# into the image. Run-time (rank step) therefore needs NO network — satisfying
# the challenge's offline, CPU-only constraint.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Keep HF/sentence-transformers caches inside the image.
    HF_HOME=/opt/hf \
    SENTENCE_TRANSFORMERS_HOME=/opt/hf \
    TRANSFORMERS_OFFLINE=0 \
    OMP_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# libgomp is required by faiss-cpu / torch OpenMP kernels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the embedding model so nothing is fetched at ranking time.
RUN python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2')"

# Application code.
COPY . .

RUN chmod +x docker/entrypoint.sh

# Defaults; override via docker-compose environment if needed.
ENV CANDIDATES_PATH=/data/candidates.jsonl \
    ARTIFACTS_DIR=/app/artifacts \
    OUT_PATH=/out/submission.csv

ENTRYPOINT ["docker/entrypoint.sh"]
