# 🏦 AI Bank Statement Automation with LLM & Personal Financial Analysis

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python)](https://www.python.org)
[![CrewAI](https://img.shields.io/badge/CrewAI-Agents-green?style=for-the-badge)](https://github.com/crewAIInc/crewAI)
[![LiteLLM](https://img.shields.io/badge/LiteLLM-Local%20%2B%20Cloud-orange?style=for-the-badge)](https://github.com/BerriAI/litellm)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> **Intelligent document automation for bank statements** — Extract, structure, analyze, and query financial data from PDFs using **YOLO + OCR + LLM**, powered by **CrewAI agents** and **local LLMs** (LM Studio / Ollama).

---

## ✨ Key Features

- **Advanced PDF Parsing** — YOLO layout detection + OCR + LLM-based table extraction
- **CrewAI Agents & Skills** — Specialized skills for bank statement parsing, PII redaction, financial analysis, and RAG querying
- **Local LLM First** — Full support for **LM Studio** and **Ollama** via LiteLLM (with async `acompletion`)
- **Reasoning Model Support** — Automatically handles models that return content in `reasoning_content`
- **Secure RAG Pipeline** — PII redaction **before** embedding into vector database (Qdrant / Chroma)
- **Financial Intelligence** — Income/expense categorization, trend analysis, and natural language querying
- **Async Ready** — Modern async LLM calls for better performance with CrewAI
- **Development Notebook** — Rich Jupyter notebook for experimentation and rapid iteration
- **MLflow Integration** — Trace and monitor LLM calls and full AI agent workflows

---

## ⚠️ Important Notes for Local LLMs (LM Studio / Ollama)

When using **local models** via LM Studio or Ollama, please follow these recommendations for best performance:

- **Model Size**: Use models with **9B parameters or larger** (e.g. `Qwen2.5-14B`, `Qwen3-27B`, `Gemma-2-9B`, `Llama-3.1-8B` or above). Smaller models (7B and below) significantly reduce output quality and reasoning ability.
- **Context Length**: Set **context size to 16K tokens or higher** (recommended **32K+**). Lower context sizes (e.g. 8K) often cause the agent to lose important instructions and produce incomplete or incorrect results.
- **JSON Output Weakness**: Local LLMs generally perform **poorly** with strict JSON/structured output. They tend to mix reasoning with JSON or fail validation.  
  **Recommendation**: Prefer asking the agent to output **clean Markdown reports** instead of forcing JSON.

> **Tip**: If you must use structured output, consider post-processing the Markdown result with a second LLM call or `instructor` library.

---

## 🛠 Tech Stack

| Component              | Technology                              |
|------------------------|-----------------------------------------|
| **LLM Orchestration**  | LiteLLM (LM Studio, Ollama, OpenAI, DeepSeek, etc.) |
| **Agent Framework**    | CrewAI + Skills                         |
| **Document Processing**| PyMuPDF, YOLO, OCR                      |
| **Vector Database**    | Qdrant, Chroma                          |
| **RAG**                | LangChain + PII redaction               |
| **Tracing & Monitoring**| MLflow                                  |
| **Frontend (Optional)**| Streamlit                               |

---

## 📁 Project Structure

```bash
AI-Bank-Statement-Document-Automation/
├── backend/
│   └── app/
│       ├── core/
│       │   └── ai_agent_skills_dev.ipynb   # ← Main development notebook
│       ├── skills/                         # CrewAI Skills
│       │   ├── bank-statement-parsing/
│       │   ├── financial-analysis/
│       │   ├── pii-handling/
│       │   └── output-formatting/
│       └── ...
├── data/
│   ├── bank-statement-document/            # Sample PDFs
│   └── uploads/
├── requirements.txt
└── README.md
```

## Multi-harness agents (experimental)

Parallel agent frameworks live under [`agents/`](agents/README.md):

- **CrewAI** — existing baseline notebook (`backend/app/core/ai_agent_skills_dev.ipynb`)
- **Deep Agents** — LangChain `deepagents` + agentskills.io (`agents/deep-agents/`)
- **Hermes** — Docker-sandboxed Hermes profile (`agents/hermes/`)

Each harness has its own skill copies. See the [design spec](docs/superpowers/specs/2026-07-20-multi-harness-agents-design.md).

## 🚀 Quick Start (Local LLM Recommended)
### 1. Setup Environment

```bash
git clone https://github.com/johnsonhk88/AI-Bank-Statement-Document-Automation-By-LLM-And-Personal-Finanical-Analysis-Prediction.git
cd AI-Bank-Statement-Document-Automation-By-LLM-And-Personal-Finanical-Analysis-Prediction

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```
## 2. Start LM Studio (Recommended)

### Open LM Studio
- Load a 9B+ model (e.g. qwen2.5-14b-instruct or qwen3-27b)
- Set Context Length to 16K or higher (32K preferred)
- Start the local server on http://localhost:1234

## 3. Run the Main Notebook
```bash
jupyter notebook backend/app/core/ai_agent_skills_dev.ipynb
```

## 🔧 Key Configuration
### Using LM Studio (Recommended)

```python
llm = LLM(
    model="openai/qwen2.5-14b-instruct",   # Use 9B+ model
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    temperature=0.6,
    max_tokens=2048,
)
```
### Async Direct Calls
```python
from litellm import acompletion

response = await acompletion(
    model="openai/your-model",
    base_url="http://localhost:1234/v1",
    api_key="lm-studio",
    messages=[{"role": "user", "content": prompt}]
)
```
## MLflow Tracing (Agent Flow Monitoring)
######  MLflow is integrated to help you trace and debug the full AI agent workflow, including LLM calls, tool usage, and task execution.

```python
import mlflow
mlflow.crewai.autolog()
mlflow.litellm.autolog()
```

-------------------------------
You can view the agent execution traces in the MLflow UI.




## 🗺 Roadmap

- Production FastAPI backend
- Multi-document collections / workspaces
- Advanced financial forecasting
- Docker + Kubernetes deployment
- Improved Streamlit dashboard with charts

## 📄 License
This project is licensed under the Apache License 2.0.

---------------------------------------

Made with ❤️ for smarter personal finance automation in Hong Kong.
⭐ If you find this project useful, please consider giving it a star!

-------------------------------------



