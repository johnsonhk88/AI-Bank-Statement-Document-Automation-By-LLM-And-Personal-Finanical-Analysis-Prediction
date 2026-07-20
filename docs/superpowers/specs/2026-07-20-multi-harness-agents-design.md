# Multi-Harness Agents Design

**Date:** 2026-07-20  
**Status:** Approved  
**Related:** Bank-statement automation, CrewAI baseline, Agent Skills (agentskills.io)

## Problem

The project uses **CrewAI** for multi-agent bank-statement workflows with `SKILL.md` packages under `backend/app/skills/`. CrewAI’s skills support is weaker than Anthropic-style / [agentskills.io](https://agentskills.io) progressive disclosure.

We need parallel agent harnesses that:

1. Keep CrewAI as a working baseline for comparison.
2. Add **Deep Agents** (LangChain `deepagents`) with first-class agentskills.io skills.
3. Add **Hermes** (Nous Research) with the same domain skills, running **inside a Docker sandbox** for secure tool/shell execution.

## Goals

- Three parallel harness folders under `agents/`.
- Per-framework **skill copies** (no shared skills root in Phase 1).
- **Separate venv** per framework to avoid dependency conflicts.
- Phase 1: scaffold + **one end-to-end path** per new harness (PDF → parse → PII → store → RAG answer).
- Hermes: **Docker terminal backend only**; host `data/` mounted **read-only**; writes only under Hermes workspace.

## Non-goals (Phase 1)

- FastAPI / production API wiring.
- Full multi-agent CrewAI parity (report agent, multi-crew orchestration).
- Single source of truth / automated skill sync across copies.
- Moving `backend/` into `agents/crewai`.
- Hermes messaging gateway (Telegram/Discord/etc.).
- Shared monorepo packaging (`uv` workspaces) — optional later.

## Decisions

| Topic | Decision |
|-------|----------|
| Layout | Approach A: `agents/{crewai,deep-agents,hermes}/` |
| CrewAI | Keep parallel; leave `backend/` intact in Phase 1 |
| Skills | Copy into each framework folder |
| Dependencies | Separate venv + requirements per framework |
| Phase 1 scope | Scaffold + one E2E path |
| Hermes isolation | Docker sandbox; `data/` RO; RW under `agents/hermes/workspace/` |

## Architecture

```
repo/
  data/                          # shared PDFs, uploads, vector stores
  backend/                       # existing CrewAI notebooks (unchanged Phase 1)
  agents/
    README.md                    # harness comparison + how to run each
    crewai/                      # thin parallel home + skill copy + pointer
    deep-agents/                 # deepagents SDK + tools + E2E runner
    hermes/                      # project profile + Docker sandbox + skills
  docs/superpowers/specs/        # this design
```

### Data flow (Phase 1 E2E)

```
sample PDF in data/
        │
        ▼
  [Harness: Deep Agents | Hermes | CrewAI notebook]
        │
        ├─ skill: bank-statement-parsing
        ├─ tool:  pdf extract / layout+OCR path
        ├─ skill: pii-handling
        ├─ tool:  pii redact
        ├─ tool:  vector store
        ├─ skill: rag-query-handling
        ├─ tool:  rag query
        └─ skill: output-format
        │
        ▼
  markdown/JSON answer + optional store under harness workspace
```

Shared **inputs** live in `data/`. Each harness may write **outputs** only in its own workspace (Hermes: `agents/hermes/workspace/`; Deep Agents: local path under `agents/deep-agents/workspace/` recommended).

## Directory layout (detailed)

### `agents/crewai/`

| Path | Role |
|------|------|
| `README.md` | Points to `backend/app/core/ai_agent_skills_dev.ipynb` as baseline |
| `skills/` | Copy of five domain skills (CrewAI-compatible frontmatter OK) |
| `requirements.txt` | Optional pin note referencing root/backend CrewAI pins |

### `agents/deep-agents/`

| Path | Role |
|------|------|
| `README.md` | Setup venv, run E2E, skills notes |
| `requirements.txt` | `deepagents`, LangChain stack, LiteLLM, vector/PDF deps |
| `.venv/` | Local venv (gitignored) |
| `skills/` | agentskills.io-normalized skill copies |
| `tools/pdf_extract.py` | PDF text/table extraction tool |
| `tools/pii_redact.py` | PII redaction tool |
| `tools/vector_store.py` | Embed + store |
| `tools/rag_query.py` | Retrieval QA |
| `agent.py` | `create_deep_agent(...)` factory |
| `run_e2e.py` | CLI: PDF → full path → answer |
| `workspace/` | Optional local outputs (gitignored contents) |

### `agents/hermes/`

| Path | Role |
|------|------|
| `README.md` | Install Hermes, Docker requirements, run E2E |
| `config.yaml` | Project terminal config: **backend docker**, limits, cwd |
| `AGENTS.md` | Project context loaded by Hermes |
| `skills/` | agentskills.io skill copies |
| `workspace/` | **RW** bind-mount inside container |
| `scripts/setup_sandbox.sh` | Verify Docker; prepare mounts |
| `scripts/run_e2e.sh` | Run sandboxed E2E against sample under `/data` |
| `docker-compose.yml` | Optional helper for sandbox image/network |

## Skills

### Source (current)

Under `backend/app/skills/`:

1. `bank-statement-parsing`
2. `financial-analysis`
3. `pii-handling`
4. `rag-query-handling`
5. `output-format`

Each is a folder with `SKILL.md` (YAML frontmatter + markdown instructions). Today they are **prompt/guideline packages**, not executable script bundles. Phase 1 preserves that model; optional `scripts/` may be added later.

### Copy policy

- Phase 1: **independent copies** under each of `agents/crewai/skills/`, `agents/deep-agents/skills/`, `agents/hermes/skills/`.
- Deep Agents and Hermes copies: frontmatter aligned to agentskills.io (`name`, `description` required). Drop CrewAI-only fields such as `compatibility: crewai>=...` on those copies.
- CrewAI copy may retain CrewAI-oriented frontmatter for the existing loader.
- Drift risk accepted; a sync script is a later enhancement, not Phase 1.

### Loading

| Harness | Load mechanism |
|---------|----------------|
| Deep Agents | `create_deep_agent(..., skills=["./skills/"], backend=FilesystemBackend(root_dir=...))` — progressive disclosure |
| Hermes | Project skills path / profile skills under `agents/hermes/skills/` |
| CrewAI | Existing `backend/app/skills/crewai_skills_loader.py` + notebook; parallel copy for comparison only |

## Deep Agents design

### Runtime

- Library: [`deepagents`](https://docs.langchain.com/oss/python/deepagents/overview) on LangGraph.
- LLM: prefer existing LiteLLM / LM Studio / Ollama patterns from the CrewAI notebook (provider-agnostic).
- Backend: `FilesystemBackend` rooted at `agents/deep-agents/` (or project root with permissions limiting writes).

### Agent factory (`agent.py`)

Responsibilities:

- Build tools list from `tools/*`.
- Attach skills directory.
- Set system prompt for bank-statement personal-finance assistant.
- Optional: subagents later; Phase 1 may use single deep agent with tools + skills.

### Tools

Thin wrappers; prefer reusing logic patterns from `backend/app/core/ai_agent_skills_dev.ipynb` without importing CrewAI:

| Tool | Behavior |
|------|----------|
| `pdf_extract` | Extract text/tables from a PDF path under `data/` |
| `pii_redact` | Redact common PII fields from extracted text/structured rows |
| `vector_store` | Chunk, embed, persist to a harness-local or shared vector path (document choice in README; prefer harness-local under `workspace/` in Phase 1 to avoid lock contention) |
| `rag_query` | Retrieve + answer using stored vectors |

### E2E entry (`run_e2e.py`)

```text
python run_e2e.py --pdf ../../data/<sample>.pdf --question "What were total debits last month?"
```

Exit 0 on successful answer print; non-zero on missing PDF/tool failure.

## Hermes design (sandboxed)

Hermes is a **CLI product**, not a thin embeddable SDK. The folder is a **project profile**: config, skills, workspace, and run scripts.

### Security requirements (mandatory)

1. **`terminal.backend: docker`** in project `config.yaml` (never `local` for this project profile).
2. Use Hermes Docker hardening defaults (cap-drop ALL, no-new-privileges, pids/memory limits).
3. **Mounts:**
   - Repo `data/` → container `/data:ro`
   - `agents/hermes/workspace/` → container `/workspace:rw`
   - `agents/hermes/skills/` → skills path (prefer read-only)
4. **`HERMES_WRITE_SAFE_ROOT=/workspace`** (add Hermes home only if required for Hermes state, not host `$HOME` broadly).
5. Secrets remain in host `~/.hermes/.env`; **minimal** `docker_forward_env` (no wholesale host env).
6. Resource limits set in config (CPU, memory, disk) appropriate for PDF/OCR workloads.
7. Document that destructive host commands are out of scope: container is the security boundary.

### E2E entry

```text
./scripts/setup_sandbox.sh   # docker available, dirs exist
./scripts/run_e2e.sh         # hermes with project config + sample task
```

Sample task text (equivalent to Deep Agents E2E): parse PDF under `/data`, redact PII, store under `/workspace`, answer one finance question, follow output-format skill.

### Dependencies

- Hermes installed per upstream docs (often managed under `~/.hermes`).
- Docker engine required on the host.
- Project folder does not vendor the full Hermes source tree.

## CrewAI parallel home

- Do **not** break existing notebook paths in Phase 1.
- `agents/crewai/README.md` documents:
  - How to run the baseline notebook.
  - That `agents/crewai/skills/` is a snapshot for side-by-side comparison with other harnesses.
- Optional later: migrate notebook into `agents/crewai/`.

## Dependency isolation

| Harness | Environment |
|---------|-------------|
| CrewAI | Existing project/root or backend venv (status quo) |
| Deep Agents | `agents/deep-agents/.venv` + `requirements.txt` |
| Hermes | Hermes managed env + Docker for execution; project scripts only |

Root `requirements.txt` is **not** required to install deepagents/hermes in Phase 1. Document activation paths in `agents/README.md`.

## Phase 1 acceptance criteria

1. `agents/` tree exists with three harness folders and top-level README.
2. Five skills copied into each harness `skills/` directory; Deep Agents/Hermes frontmatter agentskills-compatible.
3. Deep Agents: venv installable; `run_e2e.py` runs against one sample PDF (or dry-run documented if sample missing) and prints a structured answer.
4. Hermes: `config.yaml` enforces docker backend; setup/run scripts document RO `data` + RW `workspace`; E2E script invokes Hermes with the sample task.
5. CrewAI: parallel home + skill copy + pointer to existing notebook; baseline still runnable from `backend/`.
6. Design doc present under `docs/superpowers/specs/`.
7. No secrets committed; workspace outputs gitignored.

## Testing strategy

- **Deep Agents:** unit-test tool functions with fixtures (sample text, fake embeddings where possible); smoke E2E if PDF + LLM available.
- **Hermes:** script checks (docker present, config keys, mount paths); manual/sandboxed smoke E2E.
- **Skills:** validate `SKILL.md` frontmatter has `name` and `description` (simple parser test or `skills-ref` if adopted later).

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| Skill drift across three copies | Accept in Phase 1; README note; later sync script |
| Dep conflicts (CrewAI vs deepagents) | Separate venvs |
| Hermes not library-embeddable | Profile + CLI + Docker, not `import hermes` agent factory |
| Agent escapes host via Hermes local backend | Forbid `local`; docker-only project config |
| PII in vector stores | pii-handling skill + redact tool before store; workspace not committed |
| Large PDF/OCR deps in deep-agents venv | Pin minimal set; document optional YOLO path later |

## Future work (post Phase 1)

- Shared skills root or sync tooling.
- Migrate CrewAI notebook under `agents/crewai/`.
- FastAPI façade selecting harness via config.
- Deep Agents subagents matching multi-agent crew roles.
- Hermes gateway for chat ops (still sandboxed).
- `uv` workspaces / monorepo packaging.
- Executable skill `scripts/` for deterministic extract/redact.

## References

- Current CrewAI entry: `backend/app/core/ai_agent_skills_dev.ipynb`
- Current skills: `backend/app/skills/`
- Deep Agents: https://docs.langchain.com/oss/python/deepagents/overview
- Agent Skills: https://agentskills.io
- Hermes security (Docker backend): https://hermes-agent.nousresearch.com/docs/user-guide/security
