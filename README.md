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
| **LLM Orchestration** | LiteLLM (LM Studio, Ollama, OpenAI, DeepSeek, etc.) |
| **Agent Frameworks** | CrewAI + Skills · Deep Agents (LangGraph) · Hermes (Docker sandbox) |
| **Document Processing** | PyMuPDF, YOLO, OCR |
| **Vector Database** | Qdrant, Chroma |
| **RAG** | LangChain + PII redaction |
| **Tracing** | MLflow (CrewAI path) |
| **Frontend (optional)** | Streamlit |

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
├── backend/
│   └── app/
│       ├── core/
│       │   └── ai_agent_skills_dev.ipynb   # CrewAI main notebook
│       └── skills/                         # Live CrewAI skills
│           ├── bank-statement-parsing/
│           ├── financial-analysis/
│           ├── pii-handling/
│           ├── rag-query-handling/
│           └── output-format/
├── data/
│   ├── bank-statement-document/         # Sample PDFs (e.g. Dummy-Bank-Statement.pdf)
│   ├── uploads/
│   └── ...
├── docs/
│   ├── guides/
│   │   └── multi-harness-agents-guide.md   # ← Step-by-step use & test
│   └── superpowers/
│       ├── specs/                       # Design docs
│       └── plans/                       # Implementation plans
├── frontend/streamlit_app/
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
