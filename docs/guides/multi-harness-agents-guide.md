# Multi-Harness Agents — Step-by-Step Use & Test Guide

This guide walks through **setup, use, and testing** of the parallel agent harnesses under `agents/`.

| Harness | Path | Best for | LLM required? |
|---------|------|----------|----------------|
| **Deep Agents** | `agents/deep-agents/` | First try; offline E2E + tests | No (`--mode pipeline`); yes for `--mode agent` |
| **CrewAI** | `backend/` + `agents/crewai/` | Existing notebook baseline | Yes (LM Studio / Ollama / cloud) |
| **Hermes** | `agents/hermes/` | Sandboxed CLI agent (Docker) | Yes (via Hermes providers) |

**Shared inputs:** repo `data/` (sample PDFs).  
**Design:** [docs/superpowers/specs/2026-07-20-multi-harness-agents-design.md](../superpowers/specs/2026-07-20-multi-harness-agents-design.md)

---

## 1. Prerequisites

### All harnesses

- Git clone of this repository
- **Python 3.11+** (3.12 tested)
- Sample PDF present:

```text
data/bank-statement-document/Dummy-Bank-Statement.pdf
```

### Deep Agents only

- `python3 -m venv` available
- Disk space for a small local venv + Chroma under `agents/deep-agents/workspace/`

### CrewAI baseline only

- Project root venv with `pip install -r requirements.txt` (includes `crewai==1.14.7`)
- Jupyter
- Optional but recommended: **LM Studio** or **Ollama** with a **9B+** model and **16K–32K** context

### Hermes only

- [Hermes Agent CLI](https://hermes-agent.nousresearch.com/) installed and on `PATH`
- **Docker Engine** running
- Model provider configured (`hermes model` / `~/.hermes/.env`)
- Never use host `local` terminal backend for this project profile

---

## 2. Choose a harness

| Need | Use |
|------|-----|
| Verify tools/skills without an LLM | **Deep Agents** `--mode pipeline` |
| Run unit tests quickly | **Deep Agents** `pytest` |
| Full CrewAI multi-agent notebook | **CrewAI** |
| Agent with Docker isolation | **Hermes** |

**Recommended first path:** Deep Agents setup → tests → pipeline E2E (sections 3–4).

---

## 3. Deep Agents — setup

From the **repository root**:

```bash
cd agents/deep-agents
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Confirm:

```bash
python -c "import deepagents, fitz, chromadb; print('ok')"
```

> **Important:** Always set `PYTHONPATH=.` (or run from this directory with the package root on the path) so `tools` and `agent` import correctly.

---

## 4. Deep Agents — test

With `agents/deep-agents/.venv` activated:

```bash
cd agents/deep-agents
source .venv/bin/activate
PYTHONPATH=. pytest tests/ -v
```

### Expected result

```text
tests/test_skills_frontmatter.py ... PASSED
tests/test_tools.py ... PASSED
======================== 6 passed ========================
```

| Test | What it checks |
|------|----------------|
| `test_skills_have_name_and_description` (×2) | Deep Agents + Hermes skill frontmatter (`name`, `description`; no CrewAI `compatibility`) |
| `test_redact_pii_masks_email_phone_and_account_independently` | Email, phone, account redaction |
| `test_redact_pii_hyphenated_account` | Accounts like `123-456-789` |
| `test_extract_pdf_text_reads_sample` | PDF text extraction (sample or synthetic PDF) |
| `test_store_and_query_roundtrip` | Chroma store + RAG retrieve (offline hash embeddings) |

If tests fail:

1. Confirm venv is active and deps installed from `agents/deep-agents/requirements.txt` (not only root `requirements.txt`).
2. Confirm you are in `agents/deep-agents` and using `PYTHONPATH=.`.
3. Re-run a single test: `PYTHONPATH=. pytest tests/test_tools.py::test_redact_pii_hyphenated_account -v`.

---

## 5. Deep Agents — use (E2E pipeline, no LLM)

Deterministic path: **PDF extract → PII redact → vector store → RAG retrieve**.

```bash
cd agents/deep-agents
source .venv/bin/activate

PYTHONPATH=. python run_e2e.py \
  --pdf ../../data/bank-statement-document/Dummy-Bank-Statement.pdf \
  --question "What amounts appear in the statement?" \
  --mode pipeline
```

### Expected signals

- Exit code **0**
- Markdown output with headings like `# Bank Statement E2E Result`, `## Question`, `## RAG retrieval`
- Note that PII redaction was applied
- Account-style numbers in retrieved text appear as `[REDACTED_ACCOUNT]` when patterns match
- Vector data written under `agents/deep-agents/workspace/vector_stores/` (gitignored)

### CLI options

| Flag | Default | Meaning |
|------|---------|---------|
| `--pdf` | (required) | Path to bank statement PDF |
| `--question` | (required) | Question for RAG |
| `--mode` | `pipeline` | `pipeline` = tools only; `agent` = LLM deep agent |
| `--model` | `ollama:llama3.2` | Model id for `--mode agent` |

Missing PDF → exit code **2**.

### Your own PDF

```bash
PYTHONPATH=. python run_e2e.py \
  --pdf /absolute/path/to/statement.pdf \
  --question "What were total debits?" \
  --mode pipeline
```

Prefer files under `data/` so Hermes can share the same corpus later.

---

## 6. Deep Agents — optional LLM agent mode

Requires a reachable model (e.g. Ollama with `llama3.2`, or another string supported by deepagents / LangChain).

```bash
# Example: Ollama running locally
ollama pull llama3.2

cd agents/deep-agents
source .venv/bin/activate

PYTHONPATH=. python run_e2e.py \
  --pdf ../../data/bank-statement-document/Dummy-Bank-Statement.pdf \
  --question "Summarize total debits and credits" \
  --mode agent \
  --model ollama:llama3.2
```

Other model string examples (depends on your install):

- `ollama:qwen2.5:14b`
- Provider-specific ids documented by [deepagents](https://docs.langchain.com/oss/python/deepagents/overview)

**Note:** Agent mode needs network/local inference. Pipeline mode does not.

Skills are loaded from `agents/deep-agents/skills/` (agentskills.io progressive disclosure).

---

## 7. Deep Agents — troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: tools` / `agent` | Use `PYTHONPATH=.` from `agents/deep-agents` |
| `ChatOllama` / model init errors | Ensure `langchain-ollama` installed; use pipeline mode without LLM |
| Stale / mixed RAG answers | `store_documents` recreates the persist dir each run; delete `workspace/vector_stores` if needed |
| PII still shows names/addresses | Phase 1 redacts emails, phones, account-like numbers; names/addresses are not fully covered |
| Root `pip install -r requirements.txt` broke deepagents | Use **separate** `agents/deep-agents/.venv` |

---

## 8. CrewAI baseline — setup & use

CrewAI remains the original multi-agent path.

### Setup (repo root)

```bash
cd /path/to/repo
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run notebook

```bash
jupyter notebook backend/app/core/ai_agent_skills_dev.ipynb
```

### Live skills vs snapshot

| Location | Role |
|----------|------|
| `backend/app/skills/` | **Live** skills used by the notebook + `crewai_skills_loader.py` |
| `agents/crewai/skills/` | **Snapshot** for side-by-side comparison with other harnesses |

Edit live skills under `backend/app/skills/` until a future migration.

### LM Studio (recommended for CrewAI)

1. Open LM Studio, load a **9B+** model, context **16K–32K**.
2. Start server at `http://localhost:1234`.
3. In the notebook, configure CrewAI `LLM` with `base_url="http://localhost:1234/v1"` as in the root README.

### Optional smoke

```bash
python backend/app/skills/crewai_skills_loader.py
```

Should list loaded skills without error (with project venv active).

---

## 9. Hermes — setup (Docker sandbox)

### 9.1 Install Hermes CLI

Follow upstream: [Hermes Agent docs](https://hermes-agent.nousresearch.com/docs/).

Confirm:

```bash
hermes --help   # or: which hermes
docker info     # Docker must be running
```

### 9.2 Project security model

| Host path | Container | Mode |
|-----------|-----------|------|
| repo `data/` | `/data` | **read-only** |
| `agents/hermes/workspace/` | `/workspace` | read-write |
| `agents/hermes/skills/` | `/skills` | read-only |

- `agents/hermes/config.yaml` sets `terminal.backend: docker` (**never** `local`).
- Prefer `HERMES_WRITE_SAFE_ROOT=/workspace`.
- Secrets stay in host `~/.hermes/.env` (not in git).

`docker-compose.yml` documents isolation. Hermes may spawn its own container — configure mounts to match (see harness README mount checklist).

### 9.3 Validate config & host paths

```bash
cd agents/hermes
./scripts/check_config.sh
# expect: config ok: docker backend

./scripts/setup_sandbox.sh
# expect: Sandbox prerequisites OK + printed mount flags
```

Negative check (should fail):

```bash
# Do not commit this change — only for understanding the guard
grep backend config.yaml
# Must show: backend: docker
```

### 9.4 Run E2E

```bash
cd agents/hermes
./scripts/run_e2e.sh
```

| Situation | Behavior |
|-----------|----------|
| Hermes CLI installed + configured | Runs chat with project E2E prompt (flags may vary by Hermes version) |
| Hermes CLI **missing** | Still validates Docker/config; writes prompt to `workspace/e2e-prompt.txt` (exit 0 by design) |
| Sample PDF missing | Exit 2 |

Check outputs under `agents/hermes/workspace/`.

### 9.5 Hermes troubleshooting

| Symptom | Fix |
|---------|-----|
| `Docker is required` | Start Docker Engine |
| `REFUSE: terminal.backend must be docker` | Fix `config.yaml` |
| Hermes runs but cannot see PDFs | Bind-mount repo `data` → `/data:ro` in Hermes docker config |
| Writes outside workspace | Set `HERMES_WRITE_SAFE_ROOT=/workspace` |

---

## 10. Full verification checklist

Run from repo root after Deep Agents venv exists:

```bash
# --- Deep Agents ---
cd agents/deep-agents
source .venv/bin/activate
PYTHONPATH=. pytest tests/ -v
PYTHONPATH=. python run_e2e.py \
  --pdf ../../data/bank-statement-document/Dummy-Bank-Statement.pdf \
  --question "What amounts appear?" \
  --mode pipeline

# --- Hermes guards ---
cd ../hermes
./scripts/check_config.sh
./scripts/setup_sandbox.sh

# --- Skill trees exist ---
test -f ../crewai/skills/pii-handling/SKILL.md && echo crewai_skills_ok
test -f skills/bank-statement-parsing/SKILL.md && echo hermes_skills_ok
test -f ../deep-agents/skills/output-format/SKILL.md && echo deep_agents_skills_ok
```

**Pass criteria**

- [ ] Deep Agents: **6 passed**
- [ ] Deep Agents pipeline: exit **0**, markdown result printed
- [ ] Hermes: `config ok: docker backend`
- [ ] Hermes setup: paths OK (Docker available)
- [ ] All three skill trees present (5 skills each)

---

## 11. Skills reference

Each harness has its **own copy** under `skills/`. Edits **do not** auto-sync.

| Skill | Purpose |
|-------|---------|
| `bank-statement-parsing` | Balance-change rules for credit/debit; transaction structure |
| `financial-analysis` | Totals, cross-check, insights |
| `pii-handling` | Redact before vector store |
| `rag-query-handling` | Prefer RAG after store |
| `output-format` | Markdown/JSON discipline |

- Deep Agents / Hermes: agentskills.io-style frontmatter (`name`, `description`).
- CrewAI live: `backend/app/skills/` may include `compatibility: crewai...`.

---

## 12. Phase 1 limits (know these)

- Not a production FastAPI service yet.
- Deep Agents pipeline uses **offline hash embeddings** (good for smoke tests, not production semantic RAG).
- Hermes full LLM E2E needs Hermes installed and mounts aligned with the security model.
- PII redaction focuses on emails, phones, account-like numbers — not full name/address NER.
- Skill copies can drift; treat each harness tree as independent.

---

## 13. Related docs

| Doc | Path |
|-----|------|
| Agents index | [agents/README.md](../../agents/README.md) |
| Deep Agents | [agents/deep-agents/README.md](../../agents/deep-agents/README.md) |
| Hermes | [agents/hermes/README.md](../../agents/hermes/README.md) |
| CrewAI snapshot | [agents/crewai/README.md](../../agents/crewai/README.md) |
| Design spec | [docs/superpowers/specs/2026-07-20-multi-harness-agents-design.md](../superpowers/specs/2026-07-20-multi-harness-agents-design.md) |
| Implementation plan | [docs/superpowers/plans/2026-07-20-multi-harness-agents.md](../superpowers/plans/2026-07-20-multi-harness-agents.md) |
| Root README | [README.md](../../README.md) |
