FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/api/requirements.txt

# CPU-only PyTorch — ~200MB instead of ~5GB CUDA wheel
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /app/api/requirements.txt

# Presidio NER model — ~500MB, needed for name/address PII detection
RUN python -m spacy download en_core_web_lg

COPY api/ /app/api/
COPY backend/ /app/backend/
COPY data/ /app/data/
ENV PYTHONPATH=/app:/app/backend
