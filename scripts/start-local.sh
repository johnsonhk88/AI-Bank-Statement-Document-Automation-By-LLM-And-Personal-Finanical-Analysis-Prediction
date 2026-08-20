#!/usr/bin/env bash
# ============================================================
# start-local.sh — Start everything on a local PC
#
# Machine: 64 GB RAM, 12 GB VRAM
# llama.cpp server → host GPU (VRAM)
# Docker containers → CPU + RAM only
#
# Usage:
#   ./scripts/start-local.sh /path/to/model.gguf
#   ./scripts/start-local.sh /path/to/model.gguf --ctx 32768
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
INFRA_DIR="$PROJECT_DIR/infra"

# --- Arguments ---
MODEL_PATH="${1:-}"
CTX_SIZE="${3:-16384}"

if [[ -z "$MODEL_PATH" ]]; then
    echo "Usage: $0 /path/to/model.gguf [--ctx CONTEXT_SIZE]"
    echo ""
    echo "Recommended models for 12 GB VRAM:"
    echo "  - Qwen2.5-14B-Instruct-Q4_K_M.gguf  (~8 GB, good quality)"
    echo "  - Llama-3.1-8B-Instruct-Q5_K_M.gguf  (~6 GB, fast)"
    echo "  - Mistral-7B-Instruct-v0.3-Q5_K_M.gguf (~5 GB, fast)"
    echo ""
    echo "Context size: 16384 default, 32768 if model + VRAM allows"
    exit 1
fi

if [[ "$2" == "--ctx" ]] 2>/dev/null; then
    CTX_SIZE="$3"
fi

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ERROR: Model file not found: $MODEL_PATH"
    exit 1
fi

# --- Check prerequisites ---
command -v llama-server >/dev/null 2>&1 || {
    echo "ERROR: llama-server not found in PATH"
    echo "Install: https://github.com/ggerganov/llama.cpp#build"
    exit 1
}

command -v docker >/dev/null 2>&1 || {
    echo "ERROR: docker not found"
    exit 1
}

# --- Check .env exists ---
if [[ ! -f "$INFRA_DIR/.env" ]]; then
    echo "No .env found. Creating from .env.example..."
    cp "$INFRA_DIR/.env.example" "$INFRA_DIR/.env"
    echo ""
    echo "IMPORTANT: Edit $INFRA_DIR/.env before first run:"
    echo "  1. Set JWT_SECRET (run: python -c \"import secrets; print(secrets.token_urlsafe(32))\")"
    echo "  2. Set ADMIN_PASSWORD_HASH (run: python -c \"import bcrypt; print(bcrypt.hashpw(b'changeme', bcrypt.gensalt()).decode())\")"
    echo "     NOTE: Escape \$ as \$\$ in .env for Docker Compose"
    echo ""
    read -rp "Press Enter after editing .env, or Ctrl+C to abort..."
fi

# --- Stop any existing services ---
echo "==> Stopping existing containers..."
cd "$INFRA_DIR"
docker compose -f docker-compose.yml -f docker-compose.cpu.yml down 2>/dev/null || true

# --- Kill any existing llama-server ---
if pgrep -f "llama-server" >/dev/null 2>&1; then
    echo "==> Stopping existing llama-server..."
    pkill -f "llama-server" || true
    sleep 2
fi

# --- Start llama.cpp server on host GPU ---
echo "==> Starting llama.cpp server (GPU)..."
echo "    Model: $MODEL_PATH"
echo "    Context: $CTX_SIZE tokens"
echo "    Port: 8080"

llama-server \
    -m "$MODEL_PATH" \
    --port 8080 \
    --host 0.0.0.0 \
    -ngl 99 \
    --ctx-size "$CTX_SIZE" \
    --parallel 1 \
    --log-disable \
    &
LLAMA_PID=$!
echo "    PID: $LLAMA_PID"

# Wait for llama.cpp to be ready
echo "==> Waiting for llama.cpp server..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:8080/health >/dev/null 2>&1; then
        echo "    llama.cpp ready (${i}s)"
        break
    fi
    if ! kill -0 "$LLAMA_PID" 2>/dev/null; then
        echo "ERROR: llama-server exited unexpectedly"
        exit 1
    fi
    sleep 1
done

if ! curl -sf http://localhost:8080/health >/dev/null 2>&1; then
    echo "ERROR: llama.cpp server did not start within 60s"
    kill "$LLAMA_PID" 2>/dev/null || true
    exit 1
fi

# --- Start Docker containers (CPU-only) ---
echo "==> Starting Docker containers (CPU-only)..."
cd "$INFRA_DIR"
DOCKER_BUILDKIT=1 docker compose \
    -f docker-compose.yml \
    -f docker-compose.cpu.yml \
    --env-file .env \
    up -d --build

# Wait for API health
echo "==> Waiting for API..."
for i in $(seq 1 90); do
    STATUS=$(curl -sf http://localhost:8000/api/health 2>/dev/null | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
    if [[ "$STATUS" == "ok" ]]; then
        echo "    API ready (${i}s)"
        break
    fi
    sleep 1
done

HEALTH=$(curl -sf http://localhost:8000/api/health 2>/dev/null || echo '{"status":"unreachable"}')
echo ""
echo "============================================================"
echo " BankAI Local Setup"
echo "============================================================"
echo " llama.cpp : http://localhost:8080  (PID $LLAMA_PID)"
echo " API       : http://localhost:8000/api/docs"
echo " Health    : $HEALTH"
echo ""
echo " Memory budget:"
echo "   llama.cpp (GPU) : ~${CTX_SIZE} ctx on 12 GB VRAM"
echo "   Docker (CPU)    : ~17 GB capped / 64 GB available"
echo ""
echo " To stop everything:"
echo "   kill $LLAMA_PID"
echo "   cd $INFRA_DIR && docker compose -f docker-compose.yml -f docker-compose.cpu.yml down"
echo "============================================================"
