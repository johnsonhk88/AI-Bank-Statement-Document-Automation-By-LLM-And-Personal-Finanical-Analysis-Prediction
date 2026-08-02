# AI Bank Statement Automation with LLM & Personal Financial Analysis

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://www.python.org)
[![CrewAI](https://img.shields.io/badge/CrewAI-Agents-green?style=for-the-badge)](https://github.com/crewAIInc/crewAI)
[![Deep Agents](https://img.shields.io/badge/Deep%20Agents-LangChain-blue?style=for-the-badge)](https://docs.langchain.com/oss/python/deepagents/overview)
[![Hermes](https://img.shields.io/badge/Hermes-Docker%20sandbox-purple?style=for-the-badge)](https://hermes-agent.nousresearch.com/)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-Local%20%2B%20Cloud-orange?style=for-the-badge)](https://github.com/BerriAI/litellm)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **Intelligent document automation for bank statements** — Extract, structure, analyze, and query financial data from PDFs using **YOLO + OCR + LLM**, with **parallel agent harnesses** (CrewAI, Deep Agents, Hermes) and **local LLMs** (LM Studio / Ollama).

---

## Key Features

- **Advanced PDF Parsing** — YOLO layout detection + OCR + LLM-based table extraction
- **Multi-harness agents** — CrewAI baseline, LangChain **Deep Agents** (agentskills.io), **Hermes** (Docker sandbox)
- **Agent Skills** — Domain skills for bank-statement parsing, PII redaction, financial analysis, RAG, and output format
- **Local LLM First** — LM Studio and Ollama via LiteLLM (async `acompletion`)
- **Reasoning Model Support** — Handles models that return content in `reasoning_content`
- **Secure RAG Pipeline** — PII redaction **before** embedding into vector database (Qdrant / Chroma)
- **Financial Intelligence** — Income/expense categorization, trend analysis, natural language querying
- **Full-Stack API** — FastAPI + PostgreSQL + Celery + React SPA (REST auth, async document processing, agent runs)
- **GPU Acceleration** — NVIDIA GPU support for PyTorch embeddings and LLM inference (<2× build time vs CPU)
- **Development Notebook** — Jupyter notebook for CrewAI experimentation
- **MLflow Integration** — Trace LLM calls and agent workflows (CrewAI path)

---

## Important Notes for Local LLMs (LM Studio / Ollama)

When using **local models** via LM Studio or Ollama:

- **Model size**: Prefer **9B+** parameters (e.g. Qwen2.5-14B, Qwen3-27B, Gemma-2-9B, Llama-3.1-8B+). Smaller models reduce quality.
- **Context length**: **16K+** tokens (recommended **32K+**). Low context drops agent instructions.
- **JSON output**: Local models often fail strict JSON. Prefer **Markdown** reports; post-process with a second call or `instructor` if needed.

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **API** | FastAPI, SQLAlchemy 2.0 (async), Alembic, Pydantic v2 |
| **LLM Orchestration** | LiteLLM (LM Studio, Ollama, OpenAI, DeepSeek, etc.) |
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
├── agents/                              # Multi-harness agents (experimental)
│   ├── README.md
│   ├── crewai/                          # Skills snapshot + pointer to baseline
│   ├── deep-agents/                     # deepagents SDK, tools, tests, E2E
│   │   ├── run_e2e.py
│   │   ├── agent.py
│   │   ├── tools/
│   │   ├── skills/
│   │   └── tests/
│   └── hermes/                          # Docker-sandboxed Hermes profile
│       ├── config.yaml
│       ├── scripts/
│       └── skills/
├── api/                                 # FastAPI backend (full-stack v1)
│   ├── app/
│   │   ├── agents/                      # Agent adapters (CrewAI, DeepAgents)
│   │   │   ├── crewai/
│   │   │   └── registry.py
│   │   ├── api/                         # REST endpoints (auth, llm, documents, agent_runs)
│   │   ├── core/                        # Security (JWT + bcrypt), deps
│   │   ├── db/                          # SQLAlchemy session + models
│   │   ├── models/                      # SQLAlchemy ORM models
│   │   ├── schemas/                     # Pydantic v2 request/response schemas
│   │   ├── tests/                       # pytest + pytest-asyncio
│   │   ├── worker/                      # Celery tasks + runner
│   │   ├── config.py                    # pydantic-settings env loader
│   │   └── main.py                      # FastAPI application entrypoint
│   ├── alembic/                         # Database migrations
│   ├── config/                          # LLM provider catalog (YAML)
│   └── requirements.txt                 # Pinned API dependencies
├── backend/
│   └── app/
│       ├── core/
│       │   └── ai_agent_skills_dev.ipynb   # CrewAI main notebook
│       └── skills/                         # Live CrewAI skills (shared via namespace pkg)
│           ├── bank-statement-parsing/
│           ├── financial-analysis/
│           ├── pii-handling/
│           ├── rag-query-handling/
│           └── output-format/
├── infra/                               # Docker compose + Kubernetes manifests
│   ├── api.Dockerfile                   # API & Worker image
│   ├── docker-compose.yml               # PostgreSQL, Redis, Qdrant, API, Worker, Web
│   ├── docker-compose.override.yml      # Dev bind-mounts
│   ├── .env.example                     # Environment template
│   └── k8s/
├── web/                                 # React 18 + Vite 5 SPA
├── data/
│   ├── bank-statement-document/         # Sample PDFs (e.g. Dummy-Bank-Statement.pdf)
│   ├── uploads/
│   └── ...
├── docs/
│   ├── guides/
│   │   ├── multi-harness-agents-guide.md
│   │   └── full-stack-app-guide.md
│   └── superpowers/
│       ├── specs/                       # Design docs
│       └── plans/                       # Implementation plans
├── requirements.txt                     # Root / CrewAI stack
└── README.md
```

---

## Multi-harness agents (experimental)

Parallel frameworks for the same bank-statement workflow. Shared inputs live in `data/`. Each harness has its **own skill copies** (no auto-sync).

| Harness | Path | Skills | How to run |
|---------|------|--------|------------|
| **CrewAI** (baseline) | `backend/app/core/` + `agents/crewai/` | CrewAI Skills | Jupyter notebook |
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
  --pdf ../../data/bank-statement-document/Dummy-Bank-Statement.pdf \
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
jupyter notebook backend/app/core/ai_agent_skills_dev.ipynb
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
cd web && npm install && npm run dev        # frontend dev server
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
docker compose exec api python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

> **Note:** Experiment code in `backend/`, `agents/`, `frontend/streamlit_app/`, `notebooks/`, `yolo-base-layout-analysis/` is untouched by v1.

### Developer notes

- **Namespace packages:** `api/app/__init__.py` is deliberately absent — `app/` is a [PEP 420](https://peps.python.org/pep-0420/) implicit namespace package. Python automatically merges `api/app/` and `backend/app/` so `app.skills` resolves from `backend/app/skills/` without symlinks.
- **passlib / bcrypt:** The Docker image pins `bcrypt<4.1` because passlib 1.7.4 is incompatible with bcrypt ≥ 4.1 (`bcrypt.__about__` removed). If upgrading passlib, retest login flow.
- **Qdrant healthcheck** uses `/proc/net/tcp` instead of `curl` because the `qdrant/qdrant` minimal image ships without curl or wget.
- **Worker CWD:** The worker service must `cd /app/api` before starting Celery (`PYTHONPATH` alone doesn't add the api directory to Python's import search path for absolute `app.*` imports).

---

## Roadmap

- Production FastAPI backend (harness-selectable)
- Multi-document collections / workspaces
- Advanced financial forecasting
- Docker + Kubernetes deployment
- Improved Streamlit dashboard with charts
- Shared skill sync tooling across harnesses
- Stronger PII (names/addresses) and production embeddings for Deep Agents

---

## License

This project is licensed under the Apache License 2.0.

---

Made with care for smarter personal finance automation in Hong Kong.  
If you find this project useful, please consider giving it a star.
