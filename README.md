# AI Bank Statement Automation with LLM & Personal Financial Analysis

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://www.python.org)
[![CrewAI](https://img.shields.io/badge/CrewAI-Agents-green?style=for-the-badge)](https://github.com/crewAIInc/crewAI)
[![Deep Agents](https://img.shields.io/badge/Deep%20Agents-LangChain-blue?style=for-the-badge)](https://docs.langchain.com/oss/python/deepagents/overview)
[![Hermes](https://img.shields.io/badge/Hermes-Docker%20sandbox-purple?style=for-the-badge)](https://hermes-agent.nousresearch.com/)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-Local%20%2B%20Cloud-orange?style=for-the-badge)](https://github.com/BerriAI/litellm)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **Intelligent document automation for bank statements** — Extract, structure, analyze, and query financial data from PDFs using **YOLO + OCR + LLM**, with **parallel agent harnesses** (CrewAI, Deep Agents, Hermes) and **local LLMs** (vLLM, SGLang, Strata, LM Studio, Ollama).

---

## Key Features

- **Advanced PDF Parsing** — YOLO layout detection + OCR + LLM-based table extraction
- **Multi-harness agents** — CrewAI baseline, LangChain **Deep Agents** (agentskills.io), **Hermes** (Docker sandbox)
- **Agent Skills** — Domain skills for bank-statement parsing, PII redaction, financial analysis, RAG, and output format
- **Local LLM First** — vLLM, SGLang, Strata, LM Studio, Ollama via LiteLLM; frontend picks a live `/v1/models` list
- **GPU monitoring** — Prometheus + Grafana (DCGM + node-exporter), same pattern as the engine repos
- **History dashboard** — quarterly / annual totals from Postgres, Markdown + print-to-PDF
- **MCP stdio** — Claude Desktop / Cursor tools over the BankAI API
- **Reasoning Model Support** — Handles models that return content in `reasoning_content`
- **Secure RAG Pipeline** — PII redaction **before** embedding into vector database (Qdrant / Chroma)
- **Financial Intelligence** — Income/expense categorization, trend analysis, natural language querying
- **Full-Stack API** — FastAPI + PostgreSQL + Celery + React SPA (REST auth, async document processing, agent runs)
- **GPU Acceleration** — NVIDIA GPU support for PyTorch embeddings and LLM inference (<2× build time vs CPU)
- **Development Notebook** — Jupyter notebook for CrewAI experimentation
- **MLflow Integration** — Trace LLM calls and agent workflows (CrewAI path)

---

## Local LLMs (vLLM / SGLang / Strata / LM Studio / Ollama)

BankAI does not start the GPU engine. Run one of these, then select it in **New analysis**:

| Engine | GitHub | Default URL (host) | BankAI provider |
|--------|--------|--------------------|-----------------|
| **vLLM** | [docker-vllm-serving-engine](https://github.com/johnsonhk88/docker-vllm-serving-engine) | `http://127.0.0.1:8001/v1` | `vllm` |
| **SGLang** | [sglang-llm-inference-engine](https://github.com/johnsonhk88/sglang-llm-inference-engine) | `http://127.0.0.1:30000/v1` | `sglang` |
| **Strata** | [strata-llm-inference-engine](https://github.com/johnsonhk88/strata-llm-inference-engine) | `http://127.0.0.1:8080/v1` | `strata` |
| LM Studio | — | `http://127.0.0.1:1234/v1` | `lm-studio` |
| Ollama | — | `http://127.0.0.1:11434` | `ollama` |

**Port clash:** BankAI FastAPI uses **`:8000`**. Start vLLM with `--port 8001`.

When the engine is up, `GET /api/llm-models` fills the dropdown from live `/v1/models`. YAML is the fallback when the engine is down.

GPU / CPU / RAM (Prometheus + Grafana, same layout as the vLLM/SGLang `monitoring/` folders):

```bash
docker compose -f infra/monitoring/docker-compose.yml up -d
# Grafana http://localhost:3001  (admin/admin)  dashboard /d/bankai-local-llm
```

Step-by-step: **[docs/guides/local-llm-and-monitoring.md](docs/guides/local-llm-and-monitoring.md)** · MCP: **[docs/guides/mcp.md](docs/guides/mcp.md)**

Quality notes for local models:

- **Model size**: Prefer **9B+** parameters. Smaller models reduce quality.
- **Context length**: **16K+** tokens (recommended **32K+**).
- **JSON output**: Prefer **Markdown** reports; post-process if you need strict JSON.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **API** | FastAPI, SQLAlchemy 2.0 (async), Alembic, Pydantic v2 |
| **LLM Orchestration** | LiteLLM (vLLM, SGLang, Strata, LM Studio, Ollama, OpenAI, …) |
| **GPU metrics** | Prometheus + Grafana + DCGM + node-exporter (`infra/monitoring/`) |
| **Agent Frameworks** | CrewAI + Skills · Deep Agents (LangGraph) · Hermes (Docker sandbox) |
| **Document Processing** | PyMuPDF, YOLO, OCR |
| **Vector Database** | Qdrant, Chroma |
| **RAG** | LangChain + PII redaction |
| **Async Tasks** | Celery + Redis |
| **Database** | PostgreSQL 16 |
| **GPU (optional)** | NVIDIA Container Toolkit, PyTorch 2.11+cu128 |
| **Frontend** | React 18, TypeScript 5, Vite 5, TailwindCSS 3 (SPA on :80) |
| **Tracing** | MLflow (CrewAI path) |

---

## Project Structure

```text
AI-Bank-Statement-Document-Automation/
├── backend/                             # Production FastAPI service (HTTP under /api/*)
│   ├── app/                             # Python package (app.*)
│   │   ├── agents/                      # Adapters (CrewAI live; Deep Agents / Hermes stubs)
│   │   ├── api/                         # REST routers: auth, llm, documents, agent_runs
│   │   ├── core/                        # JWT + bcrypt, storage
│   │   ├── db/ models/ schemas/
│   │   ├── tests/
│   │   └── worker/                      # Celery tasks + runner
│   ├── alembic/
│   └── config/llm_providers.yaml
├── frontend/                            # Production React 18 + Vite 5 SPA
├── infra/                               # Docker compose + Dockerfiles
│   └── monitoring/                      # Prometheus + Grafana (GPU/CPU/RAM)
├── skills/                              # Canonical domain skills (edit here only)
│   ├── bank-statement-parsing/
│   ├── financial-analysis/
│   ├── pii-handling/
│   ├── rag-query-handling/
│   ├── output-format/
│   └── crewai_skills_loader.py
├── agents/                              # Experimental harnesses (each has a synced skills/ copy)
│   ├── crewai/
│   ├── deep-agents/
│   └── hermes/
├── scripts/sync_skills.py               # Copy skills/ → agents/*/skills (--apply / --check)
├── data/
│   ├── samples/Dummy-Bank-Statement.pdf
│   └── bank-statement-document/         # Extra TemplateLab samples
├── notebooks/                           # Experiments (CrewAI notebook, YOLO layout)
├── docs/guides/
└── README.md
```

**Skills policy:** edit `skills/`, then run `python scripts/sync_skills.py --apply`. Do not edit harness copies by hand. The backend loads `skills/` directly.

---

## Multi-harness agents (experimental)

Parallel frameworks for the same bank-statement workflow. Shared inputs live in `data/`. Each harness has a **generated** skill copy (`python scripts/sync_skills.py --apply`).

| Harness | Path | Skills | How to run |
|---------|------|--------|------------|
| **CrewAI** (baseline) | `notebooks/ai_agent_skills_dev.ipynb` + `agents/crewai/` | CrewAI Skills | Jupyter notebook |
| **Deep Agents** | `agents/deep-agents/` | agentskills.io | Local `.venv` + `run_e2e.py` |
| **Hermes** | `agents/hermes/` | agentskills.io | Hermes CLI + **Docker** only |

### Fastest path (no LLM)

Deep Agents deterministic pipeline — extract PDF, redact PII, store vectors, RAG answer:

```bash
cd agents/deep-agents
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -v
PYTHONPATH=. python run_e2e.py \
  --pdf ../../data/samples/Dummy-Bank-Statement.pdf \
  --question "What amounts appear in the statement?" \
  --mode pipeline
```

### Full step-by-step guide

**[docs/guides/multi-harness-agents-guide.md](docs/guides/multi-harness-agents-guide.md)** — setup, use, and test for all three harnesses, verification checklist, and troubleshooting.

Also see:

- [agents/README.md](agents/README.md) — harness index  
- [Design spec](docs/superpowers/specs/2026-07-20-multi-harness-agents-design.md)

---

## Quick Start (CrewAI + Local LLM)

### 1. Setup environment

```bash
git clone https://github.com/johnsonhk88/AI-Bank-Statement-Document-Automation-By-LLM-And-Personal-Finanical-Analysis-Prediction.git
cd AI-Bank-Statement-Document-Automation-By-LLM-And-Personal-Finanical-Analysis-Prediction

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start LM Studio (recommended)

1. Open LM Studio  
2. Load a **9B+** model (e.g. qwen2.5-14b-instruct)  
3. Set context length to **16K+** (32K preferred)  
4. Start local server at `http://localhost:1234`

### 3. Run the main CrewAI notebook

```bash
jupyter notebook notebooks/ai_agent_skills_dev.ipynb
```

### 4. Or try multi-harness agents

| Goal | Start here |
|------|------------|
| Offline tools + tests | [Guide § Deep Agents](docs/guides/multi-harness-agents-guide.md#3-deep-agents--setup) |
| CrewAI notebook | Steps 1–3 above |
| Docker-sandboxed agent | [Guide § Hermes](docs/guides/multi-harness-agents-guide.md#9-hermes--setup-docker-sandbox) |

---

## Key Configuration (CrewAI / LiteLLM)

### LM Studio

```python
llm = LLM(
    model="openai/qwen2.5-14b-instruct",   # Use 9B+ model
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    temperature=0.6,
    max_tokens=2048,
)
```

### Async direct calls

```python
from litellm import acompletion

response = await acompletion(
    model="openai/your-model",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    messages=[{"role": "user", "content": prompt}],
)
```

---

## MLflow Tracing (CrewAI path)

MLflow can trace LLM calls, tools, and CrewAI task execution:

```python
import mlflow
mlflow.crewai.autolog()
mlflow.litellm.autolog()
```

View traces in the MLflow UI after running the notebook.

---

## Full-Stack App (v1)

The full-stack app wraps the three agent harnesses (CrewAI, Deep Agents, Hermes) behind a typed FastAPI + React frontend, with PostgreSQL, Qdrant, and Celery for async background processing.

- **Design spec:** [docs/superpowers/specs/2026-07-25-full-stack-app-design.md](docs/superpowers/specs/2026-07-25-full-stack-app-design.md)
- **Implementation plan:** [docs/superpowers/plans/2026-07-25-full-stack-app.md](docs/superpowers/plans/2026-07-25-full-stack-app.md)
- **Step-by-step run & test guide:** [docs/guides/full-stack-app-guide.md](docs/guides/full-stack-app-guide.md) — prerequisites, configuration, dev/prod runs, HTTP smoke tests, pytest + vitest walkthroughs, troubleshooting, cheat sheet

### Prerequisites

- Docker 24+ with BuildKit (`DOCKER_BUILDKIT=1`)
- Docker Compose v2+
- NVIDIA Container Toolkit (for GPU support — see below)
- NVIDIA driver ≥ 580 (RTX Pro 4500 Blackwell tested; works with most RTX 20+ GPUs)

### Quickstart

```bash
# 1. Configure environment
cp infra/.env.example infra/.env

# 2. Generate JWT secret and admin password hash
python -c "import secrets; print(secrets.token_urlsafe(32))"            # → JWT_SECRET
python -c "import bcrypt; print(bcrypt.hashpw(b'your-password', bcrypt.gensalt()).decode())"  # → ADMIN_PASSWORD_HASH

# 3. Edit infra/.env — fill in JWT_SECRET and ADMIN_PASSWORD_HASH.
#    IMPORTANT: bcrypt hashes contain $ — escape them as $$ in .env:
#      ADMIN_PASSWORD_HASH=$$2b$$12$$...
#    (Docker Compose interprets $ as a variable reference.)

# 4. Start everything (prod profile: PostgreSQL + Redis + Qdrant + API + Worker + Nginx on :80)
cd infra && DOCKER_BUILDKIT=1 docker compose --profile prod up -d --build
# Open http://localhost

# 5. For development (API with reload + Vite HMR on :5173):
cd infra && docker compose up -d           # backend services
cd frontend && npm install && npm run dev   # frontend dev server
```

### GPU Configuration

To use an NVIDIA GPU inside containers:

```bash
# 1. Install NVIDIA Container Toolkit (Ubuntu/Debian)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 2. Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.2.1-base-ubuntu22.04 nvidia-smi

# 3. Build and run — the Dockerfile installs the correct GPU PyTorch wheel
#    (torch --index-url https://download.pytorch.org/whl/cu128 — matches Blackwell).
#    GPU devices are requested via deploy.resources.reservations.devices in docker-compose.yml.
cd infra && DOCKER_BUILDKIT=1 docker compose --profile prod up -d --build

# 4. Verify PyTorch sees the GPU inside the container
docker compose exec backend python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Note:** Experiments live under `notebooks/` and `agents/`. Production code is `backend/` + `frontend/` + `infra/` + `skills/`. HTTP routes stay under `/api/*`.

### Developer notes

- **`app` is a normal package:** `backend/app/__init__.py` exists. Containers set `PYTHONPATH=/app/backend:/app/skills` so `from app...` and `from crewai_skills_loader import load_skills` both work.
- **Skills:** edit `skills/` only, then `python scripts/sync_skills.py --apply`. The backend does not depend on harness copies.
- **passlib / bcrypt:** The Docker image pins `bcrypt<4.1` because passlib 1.7.4 is incompatible with bcrypt ≥ 4.1 (`bcrypt.__about__` removed). If upgrading passlib, retest login flow.
- **Qdrant healthcheck** uses `/proc/net/tcp` instead of `curl` because the `qdrant/qdrant` minimal image ships without curl or wget.
- **Worker CWD:** The worker service `cd /app/backend` then runs `celery -A app.worker.celery_app`.

---

## Roadmap

- Wire Deep Agents and Hermes adapters into the live API (CrewAI is the only enabled harness today)
- Multi-document collections / workspaces
- Kubernetes deployment
- Stronger PII (names/addresses) and production embeddings for Deep Agents

---

## License

This project is licensed under the Apache License 2.0.

---

Made with care for smarter personal finance automation in Hong Kong.  
If you find this project useful, please consider giving it a star.
