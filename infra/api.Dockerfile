FROM python:3.11-slim
WORKDIR /app

# Install build tools required for some wheels
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libpq-dev && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
# Copy requirements first to leverage Docker layer caching
COPY api/requirements.txt /app/api/requirements.txt

# Install GPU-enabled PyTorch wheel that matches the host's CUDA installation.
# Install the CUDA wheel first (adjust the index URL tag if necessary),
# then install the rest of the requirements. Use BuildKit cache mount for pip
# in a separate RUN so the mount is parsed by the Dockerfile parser (required).
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir -r /app/api/requirements.txt

# Copy code after installing deps so code changes don't bust the requirements layer
COPY api/ /app/api/
COPY backend/ /app/backend/
COPY data/ /app/data/
ENV PYTHONPATH=/app:/app/backend
