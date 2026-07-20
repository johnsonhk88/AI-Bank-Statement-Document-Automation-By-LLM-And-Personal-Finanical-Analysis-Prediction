# Multi-Harness Agents Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scaffold parallel `agents/{crewai,deep-agents,hermes}/` harnesses with per-framework skill copies, separate Deep Agents venv + E2E runner, and Docker-sandboxed Hermes E2E against shared `data/`.

**Architecture:** Three top-level harness folders keep CrewAI baseline intact under `backend/`. Deep Agents uses LangChain `deepagents` with agentskills.io progressive disclosure. Hermes runs as a project profile with `terminal.backend: docker`, RO mount of `data/`, RW `workspace/`. Skills are copied (not shared) per approved design.

**Tech Stack:** Python 3.11+, `deepagents`, LangChain/LangGraph, LiteLLM, PyMuPDF, optional Chroma/FAISS, Hermes CLI + Docker, pytest.

**Spec:** `docs/superpowers/specs/2026-07-20-multi-harness-agents-design.md`

## Global Constraints

- Do not modify or break `backend/app/core/ai_agent_skills_dev.ipynb` paths.
- Do not commit secrets, `.env`, or workspace outputs.
- Hermes project profile must never set `terminal.backend: local`.
- `data/` mounts into Hermes as **read-only**; writes only under `agents/hermes/workspace/` (and Deep Agents `workspace/`).
- Separate venv for Deep Agents; do not merge deepagents into root `requirements.txt` in Phase 1.
- Skill copies under Deep Agents/Hermes: agentskills.io frontmatter (`name`, `description` only; no `compatibility: crewai...`).
- Sample PDFs (if present): `data/bank-statement-document/Dummy-Bank-Statement.pdf`

---

## File structure (create/modify)

| Path | Responsibility |
|------|----------------|
| `agents/README.md` | How to choose/run each harness |
| `agents/crewai/README.md` | Pointer to baseline notebook + skills snapshot |
| `agents/crewai/skills/**/SKILL.md` | CrewAI-oriented skill copies |
| `agents/deep-agents/requirements.txt` | Isolated deps |
| `agents/deep-agents/skills/**/SKILL.md` | agentskills.io skill copies |
| `agents/deep-agents/tools/*.py` | pdf/pii/vector/rag tools |
| `agents/deep-agents/agent.py` | `create_deep_agent` factory |
| `agents/deep-agents/run_e2e.py` | CLI E2E |
| `agents/deep-agents/tests/test_tools.py` | Unit tests for tools |
| `agents/deep-agents/tests/test_skills_frontmatter.py` | Frontmatter validation |
| `agents/hermes/config.yaml` | Docker backend + limits |
| `agents/hermes/AGENTS.md` | Project context |
| `agents/hermes/skills/**/SKILL.md` | agentskills.io skill copies |
| `agents/hermes/scripts/setup_sandbox.sh` | Docker/mount checks |
| `agents/hermes/scripts/run_e2e.sh` | Sandboxed E2E invoke |
| `agents/hermes/docker-compose.yml` | Optional helper |
| `agents/hermes/README.md` | Install + security notes |
| `.gitignore` | Ignore harness venvs/workspaces/vector caches |
| `docs/superpowers/specs/2026-07-20-multi-harness-agents-design.md` | Already committed |

---

### Task 1: Scaffold `agents/` tree and gitignore

**Files:**
- Create: `agents/README.md`
- Create: `agents/crewai/README.md`
- Create: `agents/deep-agents/workspace/.gitkeep`
- Create: `agents/hermes/workspace/.gitkeep`
- Modify: `.gitignore`

**Interfaces:**
- Produces: directory layout for later tasks

- [ ] **Step 1: Update `.gitignore`**

Append:

```
# Multi-harness agents
agents/**/.venv/
agents/**/workspace/**
!agents/**/workspace/.gitkeep
agents/deep-agents/**/chroma_db/
agents/deep-agents/**/faiss_index/
```

- [ ] **Step 2: Create `agents/README.md`**

```markdown
# Agent harnesses

Parallel agent frameworks for bank-statement automation. Shared inputs: repo `data/`.

| Harness | Path | Skills | Runtime |
|---------|------|--------|---------|
| CrewAI (baseline) | `crewai/` + `backend/app/core/` | CrewAI Skills | existing project venv + notebook |
| Deep Agents | `deep-agents/` | agentskills.io | local `.venv` + `run_e2e.py` |
| Hermes | `hermes/` | agentskills.io | Hermes CLI + **Docker sandbox** |

## Quick start

### CrewAI baseline
See `crewai/README.md` and run `backend/app/core/ai_agent_skills_dev.ipynb`.

### Deep Agents
```bash
cd agents/deep-agents
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_e2e.py --pdf ../../data/bank-statement-document/Dummy-Bank-Statement.pdf --question "Summarize total debits and credits"
```

### Hermes (Docker required)
```bash
cd agents/hermes
./scripts/setup_sandbox.sh
./scripts/run_e2e.sh
```

## Skills policy
Each harness has its **own copy** of domain skills under `skills/`. Edits do not auto-sync.
```

- [ ] **Step 3: Create `agents/crewai/README.md`**

```markdown
# CrewAI harness (baseline)

Phase 1 keeps the working baseline in:

- Notebook: `backend/app/core/ai_agent_skills_dev.ipynb`
- Skills (live): `backend/app/skills/`
- Loader: `backend/app/skills/crewai_skills_loader.py`

This folder holds a **skills snapshot** under `skills/` for side-by-side comparison with Deep Agents and Hermes. Prefer editing `backend/app/skills/` for the live CrewAI path until a later migration.

## Run baseline

From repo root, with the project venv that has `crewai==1.14.7`:

```bash
jupyter notebook backend/app/core/ai_agent_skills_dev.ipynb
```
```

- [ ] **Step 4: Create workspace placeholders**

```bash
mkdir -p agents/deep-agents/workspace agents/hermes/workspace
touch agents/deep-agents/workspace/.gitkeep agents/hermes/workspace/.gitkeep
```

- [ ] **Step 5: Commit**

```bash
git add .gitignore agents/README.md agents/crewai/README.md agents/deep-agents/workspace/.gitkeep agents/hermes/workspace/.gitkeep
git commit -m "chore: scaffold agents/ multi-harness layout and gitignore"
```

---

### Task 2: Copy and normalize skills into three harnesses

**Files:**
- Create: `agents/crewai/skills/{bank-statement-parsing,financial-analysis,pii-handling,rag-query-handling,output-format}/SKILL.md`
- Create: `agents/deep-agents/skills/.../SKILL.md`
- Create: `agents/hermes/skills/.../SKILL.md`
- Create: `agents/deep-agents/tests/test_skills_frontmatter.py`

**Interfaces:**
- Consumes: `backend/app/skills/*/SKILL.md`
- Produces: normalized skill trees; test asserts `name` + `description` on deep-agents and hermes

- [ ] **Step 1: Write failing frontmatter test**

Create `agents/deep-agents/tests/test_skills_frontmatter.py`:

```python
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKILL_ROOTS = [
    ROOT / "skills",
    ROOT.parent / "hermes" / "skills",
]


def _parse_frontmatter(text: str) -> dict[str, str]:
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", text, re.DOTALL)
    assert m, "missing YAML frontmatter"
    data = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        data[k.strip()] = v.strip()
    return data


@pytest.mark.parametrize("skills_dir", SKILL_ROOTS)
def test_skills_have_name_and_description(skills_dir: Path):
    assert skills_dir.is_dir(), f"missing {skills_dir}"
    skill_files = list(skills_dir.glob("*/SKILL.md"))
    assert len(skill_files) == 5
    for path in skill_files:
        meta = _parse_frontmatter(path.read_text(encoding="utf-8"))
        assert meta.get("name"), f"{path} missing name"
        assert meta.get("description"), f"{path} missing description"
        assert "compatibility" not in meta, f"{path} must not use crewai compatibility field"
```

- [ ] **Step 2: Run test — expect FAIL (missing dirs)**

```bash
cd agents/deep-agents && python -m pytest tests/test_skills_frontmatter.py -v
```

Expected: FAIL (skills missing or pytest not installed — install pytest in next task if needed; for this step `pip install pytest` in a temp way or use system python).

- [ ] **Step 3: Copy skills with a scripted normalize**

From repo root:

```bash
SKILLS="bank-statement-parsing financial-analysis pii-handling rag-query-handling output-format"
for target in agents/crewai/skills agents/deep-agents/skills agents/hermes/skills; do
  mkdir -p "$target"
  for s in $SKILLS; do
    mkdir -p "$target/$s"
    cp "backend/app/skills/$s/SKILL.md" "$target/$s/SKILL.md"
  done
done
```

For **deep-agents** and **hermes** only, strip `compatibility:` lines and fix `output-format` name trailing space:

```bash
python3 <<'PY'
from pathlib import Path
import re
for root in [Path("agents/deep-agents/skills"), Path("agents/hermes/skills")]:
    for path in root.glob("*/SKILL.md"):
        text = path.read_text(encoding="utf-8")
        text = re.sub(r"^compatibility:.*\n", "", text, flags=re.M)
        text = re.sub(r"^name:\s*output-format\s*$", "name: output-format", text, flags=re.M)
        # tool name hints for non-CrewAI harnesses
        text = text.replace("`PDF Extractor`", "`pdf_extract`")
        text = text.replace("`pii_redaction_tool`", "`pii_redact`")
        text = text.replace("`rag_tool`", "`rag_query`")
        path.write_text(text, encoding="utf-8")
print("normalized")
PY
```

Leave `agents/crewai/skills` as faithful copies (may keep `compatibility`).

- [ ] **Step 4: Run frontmatter test — expect PASS**

```bash
pip install pytest -q
cd /path/to/repo
python -m pytest agents/deep-agents/tests/test_skills_frontmatter.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add agents/crewai/skills agents/deep-agents/skills agents/hermes/skills agents/deep-agents/tests/test_skills_frontmatter.py
git commit -m "feat: copy domain skills into crewai, deep-agents, and hermes harnesses"
```

---

### Task 3: Deep Agents tools (TDD)

**Files:**
- Create: `agents/deep-agents/tools/__init__.py`
- Create: `agents/deep-agents/tools/pdf_extract.py`
- Create: `agents/deep-agents/tools/pii_redact.py`
- Create: `agents/deep-agents/tools/vector_store.py`
- Create: `agents/deep-agents/tools/rag_query.py`
- Create: `agents/deep-agents/tests/test_tools.py`
- Create: `agents/deep-agents/requirements.txt`

**Interfaces:**
- Produces:
  - `extract_pdf_text(path: str | Path) -> str`
  - `redact_pii(text: str) -> str`
  - `store_documents(texts: list[str], persist_dir: str | Path) -> str` → returns persist_dir
  - `query_store(question: str, persist_dir: str | Path, k: int = 4) -> str`

- [ ] **Step 1: Write `requirements.txt`**

```text
# agents/deep-agents — isolated from root CrewAI pins
deepagents>=0.2.0
langchain>=0.3.25
langchain-community>=0.3.24
langchain-text-splitters>=0.3.8
langchain-chroma>=0.2.6
chromadb>=1.0.20
litellm>=1.79.2
pymupdf>=1.26.0
python-dotenv>=1.0.0
pydantic>=2.0
pytest>=8.0
```

- [ ] **Step 2: Write failing tool tests**

`agents/deep-agents/tests/test_tools.py`:

```python
from pathlib import Path

from tools.pii_redact import redact_pii
from tools.pdf_extract import extract_pdf_text
from tools.vector_store import store_documents
from tools.rag_query import query_store


def test_redact_pii_masks_email_and_account():
    raw = "Email jane.doe@example.com account 12345678 phone +1-555-0100"
    out = redact_pii(raw)
    assert "jane.doe@example.com" not in out
    assert "12345678" not in out
    assert "[REDACTED" in out or "***" in out


def test_extract_pdf_text_reads_sample(tmp_path: Path):
    sample = Path(__file__).resolve().parents[3] / "data" / "bank-statement-document" / "Dummy-Bank-Statement.pdf"
    if not sample.exists():
        # minimal synthetic pdf via pymupdf
        import fitz
        sample = tmp_path / "mini.pdf"
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Opening Balance 100\nCoffee 5.00 Balance 95")
        doc.save(sample)
        doc.close()
    text = extract_pdf_text(sample)
    assert isinstance(text, str)
    assert len(text.strip()) > 0


def test_store_and_query_roundtrip(tmp_path: Path):
    persist = tmp_path / "vs"
    store_documents(
        ["Total debits were 250.00 dollars for groceries and rent."],
        persist_dir=persist,
    )
    answer = query_store("What were total debits?", persist_dir=persist, k=2)
    assert isinstance(answer, str)
    assert len(answer) > 0
```

- [ ] **Step 3: Run tests — expect FAIL**

```bash
cd agents/deep-agents
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest tests/test_tools.py -v
```

Expected: FAIL import errors

- [ ] **Step 4: Implement tools**

`tools/__init__.py`:

```python
from tools.pdf_extract import extract_pdf_text
from tools.pii_redact import redact_pii
from tools.vector_store import store_documents
from tools.rag_query import query_store

__all__ = [
    "extract_pdf_text",
    "redact_pii",
    "store_documents",
    "query_store",
]
```

`tools/pii_redact.py`:

```python
import re

_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
    (re.compile(r"\b\d{8,17}\b"), "[REDACTED_ACCOUNT]"),
]


def redact_pii(text: str) -> str:
    out = text
    for pattern, repl in _PATTERNS:
        out = pattern.sub(repl, out)
    return out
```

`tools/pdf_extract.py`:

```python
from pathlib import Path

import fitz


def extract_pdf_text(path: str | Path) -> str:
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    try:
        parts = [page.get_text("text") for page in doc]
    finally:
        doc.close()
    return "\n".join(parts).strip()
```

`tools/vector_store.py`:

```python
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class HashEmbedding(Embeddings):
    """Offline deterministic embedding for tests/local smoke without API keys."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for i, ch in enumerate(text.encode("utf-8")):
            vec[i % self.dim] += (ch % 31) / 31.0
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def store_documents(texts: list[str], persist_dir: str | Path) -> str:
    path = Path(persist_dir)
    path.mkdir(parents=True, exist_ok=True)
    docs = [Document(page_content=t) for t in texts if t and t.strip()]
    if not docs:
        raise ValueError("no documents to store")
    Chroma.from_documents(
        documents=docs,
        embedding=HashEmbedding(),
        persist_directory=str(path),
    )
    return str(path.resolve())
```

`tools/rag_query.py`:

```python
from pathlib import Path

from langchain_chroma import Chroma

from tools.vector_store import HashEmbedding


def query_store(question: str, persist_dir: str | Path, k: int = 4) -> str:
    path = Path(persist_dir)
    if not path.exists():
        raise FileNotFoundError(f"vector store not found: {path}")
    vs = Chroma(persist_directory=str(path), embedding_function=HashEmbedding())
    docs = vs.similarity_search(question, k=k)
    if not docs:
        return "No relevant documents found."
    chunks = "\n---\n".join(d.page_content for d in docs)
    return f"Question: {question}\n\nRetrieved context:\n{chunks}"
```

- [ ] **Step 5: Run tests — expect PASS**

```bash
cd agents/deep-agents && source .venv/bin/activate
PYTHONPATH=. pytest tests/test_tools.py tests/test_skills_frontmatter.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add agents/deep-agents/requirements.txt agents/deep-agents/tools agents/deep-agents/tests/test_tools.py
git commit -m "feat(deep-agents): add pdf, pii, vector, and rag tools with tests"
```

---

### Task 4: Deep Agents agent factory + E2E runner

**Files:**
- Create: `agents/deep-agents/agent.py`
- Create: `agents/deep-agents/run_e2e.py`
- Create: `agents/deep-agents/README.md`

**Interfaces:**
- Produces: `build_agent()` → runnable deep agent
- Produces: `run_pipeline(pdf_path, question) -> str` (deterministic tool chain; agent optional when no LLM)

- [ ] **Step 1: Implement deterministic pipeline + optional agent**

`agent.py`:

```python
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend

from tools.pdf_extract import extract_pdf_text
from tools.pii_redact import redact_pii
from tools.vector_store import store_documents
from tools.rag_query import query_store

ROOT = Path(__file__).resolve().parent
SKILLS = str(ROOT / "skills")
WORKSPACE = ROOT / "workspace"


def pdf_extract(path: str) -> str:
    """Extract text from a bank-statement PDF path."""
    return extract_pdf_text(path)


def pii_redact(text: str) -> str:
    """Redact PII from text before storage or display."""
    return redact_pii(text)


def vector_store(text: str, collection_name: str = "statements") -> str:
    """Store redacted text into the local vector DB; returns persist path."""
    persist = WORKSPACE / "vector_stores" / collection_name
    return store_documents([text], persist_dir=persist)


def rag_query(question: str, collection_name: str = "statements") -> str:
    """Answer a question using RAG over stored statement chunks."""
    persist = WORKSPACE / "vector_stores" / collection_name
    return query_store(question, persist_dir=persist)


def build_agent(model: str = "ollama:llama3.2"):
    """Create a deep agent with bank-statement tools and skills.

    Requires a configured LLM provider. For offline smoke tests use
    `run_pipeline` instead.
    """
    backend = FilesystemBackend(root_dir=str(ROOT))
    return create_deep_agent(
        model=model,
        tools=[pdf_extract, pii_redact, vector_store, rag_query],
        skills=[SKILLS],
        backend=backend,
        system_prompt=(
            "You are a bank-statement automation assistant. "
            "Follow loaded skills. Always extract PDF, redact PII before vector_store, "
            "then answer with rag_query. Prefer structured markdown output."
        ),
    )


def run_pipeline(pdf_path: str | Path, question: str, collection_name: str = "statements") -> str:
    """Deterministic E2E path without an LLM (tools only)."""
    raw = extract_pdf_text(pdf_path)
    clean = redact_pii(raw)
    store_documents([clean], persist_dir=WORKSPACE / "vector_stores" / collection_name)
    retrieval = query_store(question, persist_dir=WORKSPACE / "vector_stores" / collection_name)
    return (
        "# Bank Statement E2E Result\n\n"
        f"## Question\n{question}\n\n"
        f"## RAG retrieval\n{retrieval}\n\n"
        "## Notes\nPII redaction applied before vector store.\n"
    )
```

`run_e2e.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent import run_pipeline, build_agent


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Deep Agents bank-statement E2E")
    parser.add_argument("--pdf", required=True, type=Path, help="Path to bank statement PDF")
    parser.add_argument("--question", required=True, help="Finance question for RAG")
    parser.add_argument(
        "--mode",
        choices=("pipeline", "agent"),
        default="pipeline",
        help="pipeline=deterministic tools only; agent=deepagents LLM loop",
    )
    parser.add_argument("--model", default="ollama:llama3.2", help="Model id for --mode agent")
    args = parser.parse_args(argv)

    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        return 2

    if args.mode == "pipeline":
        print(run_pipeline(args.pdf, args.question))
        return 0

    agent = build_agent(model=args.model)
    prompt = (
        f"Process bank statement PDF at {args.pdf.resolve()}. "
        f"Extract text, redact PII, store vectors, then answer: {args.question}"
    )
    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
    messages = result.get("messages") if isinstance(result, dict) else None
    if messages:
        print(messages[-1].content if hasattr(messages[-1], "content") else messages[-1])
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

`README.md`:

```markdown
# Deep Agents harness

LangChain **deepagents** with agentskills.io skills for bank-statement automation.

## Setup

```bash
cd agents/deep-agents
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## E2E (no LLM — deterministic tools)

```bash
PYTHONPATH=. python run_e2e.py \
  --pdf ../../data/bank-statement-document/Dummy-Bank-Statement.pdf \
  --question "What amounts appear in the statement?" \
  --mode pipeline
```

## E2E (LLM agent)

Requires a local/remote model reachable by the deepagents model string:

```bash
PYTHONPATH=. python run_e2e.py \
  --pdf ../../data/bank-statement-document/Dummy-Bank-Statement.pdf \
  --question "Summarize total debits and credits" \
  --mode agent \
  --model ollama:llama3.2
```

Skills live in `./skills/`. Outputs go to `./workspace/` (gitignored).
```

- [ ] **Step 2: Smoke-run pipeline mode**

```bash
cd agents/deep-agents && source .venv/bin/activate
PYTHONPATH=. python run_e2e.py \
  --pdf ../../data/bank-statement-document/Dummy-Bank-Statement.pdf \
  --question "What amounts appear?" \
  --mode pipeline
```

Expected: markdown printed; exit 0

- [ ] **Step 3: Commit**

```bash
git add agents/deep-agents/agent.py agents/deep-agents/run_e2e.py agents/deep-agents/README.md
git commit -m "feat(deep-agents): add agent factory and E2E runner"
```

---

### Task 5: Hermes sandboxed profile + scripts

**Files:**
- Create: `agents/hermes/config.yaml`
- Create: `agents/hermes/AGENTS.md`
- Create: `agents/hermes/docker-compose.yml`
- Create: `agents/hermes/scripts/setup_sandbox.sh`
- Create: `agents/hermes/scripts/run_e2e.sh`
- Create: `agents/hermes/README.md`
- Create: `agents/hermes/scripts/check_config.sh`

**Interfaces:**
- Produces: docker-only Hermes project config; setup/run scripts exit non-zero if docker missing or backend != docker

- [ ] **Step 1: Write config and project context**

`config.yaml`:

```yaml
# Project-local Hermes config fragment for bank-statement automation.
# Merge/copy into the active Hermes profile or point HERMES config at this file
# per upstream docs. DO NOT use terminal.backend: local in this project.

terminal:
  backend: docker
  cwd: /workspace
  container_cpu: 2
  container_memory: 4096
  container_disk: 20480
  container_persistent: true
  docker_forward_env: []
  # Optional explicit image; override if your environment needs OCR/python preinstalled
  # docker_image: "nikolaik/python-nodejs:python3.11-nodejs20"

approvals:
  mode: smart
  timeout: 60

security:
  allow_private_urls: false
```

`AGENTS.md`:

```markdown
# Bank Statement Automation (Hermes)

You automate personal bank-statement analysis for this repository.

## Paths inside the sandbox
- Read-only statements: `/data/` (host repo `data/`)
- Writable outputs: `/workspace/` only
- Skills: project skills for parsing, PII, RAG, analysis, output format

## Mandatory workflow
1. Read PDF under `/data/` (never leave sandbox mounts).
2. Follow **bank-statement-parsing** skill (balance-change rules for credit/debit).
3. Follow **pii-handling** — redact before any long-term store under `/workspace/`.
4. Store artifacts only under `/workspace/`.
5. Answer questions using stored redacted content; follow **output-format**.

## Security
- Do not attempt to access host paths outside `/data` and `/workspace`.
- Do not print secrets or raw account numbers.
```

`docker-compose.yml`:

```yaml
# Optional helper to pre-pull/run a compatible sandbox image.
# Hermes manages its own docker backend; this file documents the isolation model.
services:
  hermes-sandbox:
    image: nikolaik/python-nodejs:python3.11-nodejs20
    working_dir: /workspace
    volumes:
      - ../../data:/data:ro
      - ./workspace:/workspace:rw
      - ./skills:/skills:ro
    environment:
      - HERMES_WRITE_SAFE_ROOT=/workspace
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true
    pids_limit: 256
    mem_limit: 4g
    cpus: 2.0
    command: ["sleep", "infinity"]
```

- [ ] **Step 2: Write setup and check scripts**

`scripts/check_config.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$ROOT/config.yaml"
if [[ ! -f "$CFG" ]]; then
  echo "Missing config: $CFG" >&2
  exit 1
fi
if grep -E '^\s*backend:\s*local\s*$' "$CFG" >/dev/null; then
  echo "REFUSE: terminal.backend must not be local" >&2
  exit 1
fi
if ! grep -E '^\s*backend:\s*docker\s*$' "$CFG" >/dev/null; then
  echo "REFUSE: terminal.backend must be docker" >&2
  exit 1
fi
echo "config ok: docker backend"
```

`scripts/setup_sandbox.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"

"$ROOT/scripts/check_config.sh"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required for Hermes sandbox" >&2
  exit 1
fi
docker info >/dev/null

mkdir -p "$ROOT/workspace"
if [[ ! -d "$REPO/data" ]]; then
  echo "Missing repo data/: $REPO/data" >&2
  exit 1
fi

# Document env expected at runtime
cat > "$ROOT/workspace/.sandbox-env.example" <<EOF
# Export on host before run_e2e.sh (do not commit real secrets)
# export HERMES_WRITE_SAFE_ROOT=/workspace
# Ensure Hermes uses project config with terminal.backend=docker
EOF

echo "Sandbox prerequisites OK"
echo "  data (RO):  $REPO/data -> /data"
echo "  workspace:  $ROOT/workspace -> /workspace"
echo "  skills:     $ROOT/skills"
```

`scripts/run_e2e.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
PDF_REL="bank-statement-document/Dummy-Bank-Statement.pdf"
PDF_HOST="$REPO/data/$PDF_REL"

"$ROOT/scripts/setup_sandbox.sh"

if [[ ! -f "$PDF_HOST" ]]; then
  echo "Sample PDF missing: $PDF_HOST" >&2
  exit 2
fi

PROMPT=$(cat <<EOF
Run the bank-statement E2E inside the sandbox:
1) Read PDF at /data/${PDF_REL}
2) Parse using bank-statement-parsing skill
3) Redact PII using pii-handling skill
4) Write redacted extract to /workspace/e2e-extract.md
5) Answer: What amounts and balances appear in the statement?
6) Write the final answer only as markdown to /workspace/e2e-answer.md
Follow output-format skill. Do not access paths outside /data and /workspace.
EOF
)

if ! command -v hermes >/dev/null 2>&1; then
  echo "hermes CLI not found on PATH."
  echo "Install from https://hermes-agent.nousresearch.com/ then re-run."
  echo "Documented prompt saved for manual run:"
  printf '%s\n' "$PROMPT" | tee "$ROOT/workspace/e2e-prompt.txt"
  exit 0
fi

# Prefer project config; exact flag names may vary by Hermes version — adjust per `hermes --help`.
export HERMES_WRITE_SAFE_ROOT="${HERMES_WRITE_SAFE_ROOT:-/workspace}"
set +e
hermes chat --config "$ROOT/config.yaml" -q "$PROMPT"
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "hermes chat failed (exit $status). Prompt is in workspace/e2e-prompt.txt for manual retry."
  printf '%s\n' "$PROMPT" > "$ROOT/workspace/e2e-prompt.txt"
  exit $status
fi
echo "Hermes E2E finished. Check $ROOT/workspace/"
```

Make executable:

```bash
chmod +x agents/hermes/scripts/*.sh
```

- [ ] **Step 3: Write Hermes README**

```markdown
# Hermes harness (Docker sandbox)

Nous Research **Hermes Agent** profile for this repo. Tool/shell execution must use **Docker**, not the host shell.

## Security model

| Mount | Container path | Mode |
|-----------------------|------|
| repo `data/` | `/data` | **read-only** |
| `agents/hermes/workspace/` | `/workspace` | read-write |
| `agents/hermes/skills/` | skills path | read-only preferred |

- `config.yaml` sets `terminal.backend: docker` (enforced by `scripts/check_config.sh`).
- `HERMES_WRITE_SAFE_ROOT=/workspace`
- No secrets in git; use host `~/.hermes/.env`.

## Prerequisites

1. Install Hermes CLI (upstream installer).
2. Docker Engine running.
3. Model provider configured via `hermes model` / env.

## Run

```bash
cd agents/hermes
./scripts/setup_sandbox.sh
./scripts/run_e2e.sh
```

If `hermes` is not installed, `run_e2e.sh` still validates Docker/config and writes the E2E prompt under `workspace/`.

## Skills

Copied under `./skills/` (agentskills.io). Independent from CrewAI/Deep Agents copies.
```

- [ ] **Step 4: Verify config guard**

```bash
agents/hermes/scripts/check_config.sh
# expect: config ok: docker backend

# negative test
cp agents/hermes/config.yaml /tmp/hermes-bad.yaml
# temporarily break — or:
bash -c 'grep -q "backend: docker" agents/hermes/config.yaml && echo positive_ok'
```

- [ ] **Step 5: Run setup script**

```bash
agents/hermes/scripts/setup_sandbox.sh
```

Expected: Sandbox prerequisites OK (if Docker available)

- [ ] **Step 6: Commit**

```bash
git add agents/hermes
git commit -m "feat(hermes): add Docker-sandboxed project profile and E2E scripts"
```

---

### Task 6: Wire CrewAI skill snapshot note + final verification

**Files:**
- Modify: `agents/crewai/README.md` (if needed after skills copy)
- Modify: root `README.md` (short pointer only)

**Interfaces:**
- Produces: discoverability from root README

- [ ] **Step 1: Add a short section to root `README.md`**

After the existing agent/CrewAI section (or near project structure), add:

```markdown
## Multi-harness agents (experimental)

Parallel agent frameworks live under [`agents/`](agents/README.md):

- **CrewAI** — existing baseline notebook (`backend/app/core/ai_agent_skills_dev.ipynb`)
- **Deep Agents** — LangChain `deepagents` + agentskills.io (`agents/deep-agents/`)
- **Hermes** — Docker-sandboxed Hermes profile (`agents/hermes/`)

Each harness has its own skill copies. See the [design spec](docs/superpowers/specs/2026-07-20-multi-harness-agents-design.md).
```

- [ ] **Step 2: Full verification checklist**

```bash
# skills frontmatter + tools
cd agents/deep-agents && source .venv/bin/activate
PYTHONPATH=. pytest tests/ -v
PYTHONPATH=. python run_e2e.py --pdf ../../data/bank-statement-document/Dummy-Bank-Statement.pdf --question "What amounts appear?" --mode pipeline

# hermes guards
cd ../hermes && ./scripts/check_config.sh && ./scripts/setup_sandbox.sh

# tree sanity
test -f ../crewai/skills/pii-handling/SKILL.md
test -f skills/bank-statement-parsing/SKILL.md
test -f ../deep-agents/skills/output-format/SKILL.md
```

Expected: all green (Hermes E2E LLM optional if CLI missing)

- [ ] **Step 3: Commit**

```bash
git add README.md agents/crewai/README.md
git commit -m "docs: link multi-harness agents from root README"
```

---

## Self-review vs spec

| Spec requirement | Task |
|------------------|------|
| `agents/{crewai,deep-agents,hermes}/` | Task 1 |
| Per-framework skill copies | Task 2 |
| Separate Deep Agents venv | Task 3–4 |
| Deep Agents E2E PDF→PII→store→RAG | Task 3–4 (`run_pipeline`) |
| Hermes docker-only + RO data + RW workspace | Task 5 |
| CrewAI baseline preserved | Task 1, 6 (no backend move) |
| gitignore workspaces/venvs | Task 1 |
| Acceptance criteria | Task 6 verification |

**Placeholder scan:** none intentional. Hermes CLI flags (`hermes chat --config`) may need adjustment per installed Hermes version — scripts fall back to writing the prompt file.

**Type consistency:** tool names `pdf_extract`, `pii_redact`, `vector_store`, `rag_query` match skill text after normalization and `agent.py` exports.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-07-20-multi-harness-agents.md`.

**Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks  
2. **Inline Execution** — execute tasks in this session with executing-plans checkpoints  

Which approach?
