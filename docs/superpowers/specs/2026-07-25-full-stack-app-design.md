# Full-Stack App Design (Bank-Statement Agent Wrapper)

**Date:** 2026-07-25
**Status:** Approved
**Related:**
- `docs/superpowers/specs/2026-07-20-multi-harness-agents-design.md` (Approved) — multi-harness design (CrewAI / Deep Agents / Hermes) this app wraps
- `backend/app/core/ai_agent_skills_dev.ipynb` — CrewAI baseline ported by `api/agents/crewai/adapter.py`
- `README.md` roadmap items: "Production FastAPI backend (harness-selectable)", "Improved Streamlit dashboard with charts", "Docker + Kubernetes deployment", "Multi-document collections / workspaces"

## Problem

The project currently consists of prototypes and one runnable, tested agent path (Deep Agents). There is no web service, no relational DB, no task queue, no build frontend. The CrewAI baseline lives in a Jupyter notebook and cannot be invoked from a UI. Multi-PDF batch analysis requires manually re-running the notebook per file. LLM/provider choice is hard-coded per prototype.

We need a production full-stack application that:
- Wraps the existing three agent harnesses behind a typed service interface.
- Lets a user upload multiple bank-statement PDFs, ask one shared question, and view structured Markdown reports + extracted transactions per PDF.
- Persists jobs, documents, and results in Postgres.
- Runs long agent work in a background queue (Redis + Celery) so the API stays responsive.
- Standardizes on a single production VectorDB (Qdrant).
- Lets the user select LLM provider and model per run (local via LM Studio/Ollama, or cloud via OpenRouter/Gemini/openai).
- Replaces the Streamlit prototypes with a React + Vite SPA.
- Keeps all existing experiment code (`backend/`, `agents/`, `frontend/streamlit_app/`, `notebooks/`, `yolo-base-layout-analysis/`) untouched.

## Goals

- New top-level `api/` (FastAPI async service), `web/` (React+Vite SPA), `infra/` (docker-compose + Dockerfiles + future k8s stub) added as siblings — experiment code untouched.
- SQLAlchemy 2.0 typed ORM + Alembic schema versioning on Postgres.
- Redis + Celery (prefork pool) for batch agent jobs; one Celery task per PDF per batch.
- Qdrant as the production VectorDB; per-document collections (`doc_<id>`) so reruns reuse embeddings.
- LLM provider/model selector: config-driven YAML catalog resolved at runtime via LiteLLM.
- CrewAI adapter implemented first; AgentRegistry structured so Deep Agents and Hermes adapters drop in later without touching the API or UI.
- Frontend: agent dropdown (CrewAI enabled; Deep Agents/Hermes disabled "coming soon"), grouped model dropdown (Local/Cloud), multi-PDF upload, shared question, batch list + detail pages with per-item cards, Markdown viewer, transactions table, JWT login, polling progress every ~2s.
- Docker-compose with `dev` and `prod` profiles; prod serves the built SPA via Nginx on :80 and reverse-proxies `/api/*` to FastAPI.

## Non-goals (v1)

- Multi-user accounts / signup flow (single env-seeded admin only; `users` table exists with one row).
- Per-user workspaces / multi-tenant isolation.
- Forecasting / anomaly detection / spending-category charts (deferred to v2 dashboard — the "Prediction" in the project name lands later).
- PostgreSQL → pgvector consolidation (Qdrant remains the vector store).
- DB-backed LLM catalog (YAML config is catalog of record for v1).
- Streaming agent progress (polling every ~2s is sufficient at v1's scope).
- Per-item model override; one model per batch.
- Hard delete of documents/runs (soft delete via `deleted_at` only).
- Kubernetes manifests (compose first; `infra/k8s/` is a stub for a later follow-up; user flagged microk8s as the eventual target).
- Wiring Deep Agents or Hermes into the live API (registry entries registered as `enabled=False`; adapters land in v1.1).

## Decisions

| Topic | Decision |
|-------|----------|
| Architecture approach | Approach A — port CrewAI notebook logic into a clean adapter module under `api/agents/crewai/` |
| v1 scope | Thin agent wrapper (upload → ask → view Markdown report + transactions) |
| Auth | Single env-seeded admin; `users` table exists with one row; JWT, 15-min TTL |
| VectorDB | Qdrant (Docker service); per-document collections `doc_<id>`; reruns reuse embeddings |
| Topology | Docker-compose first; profiles `dev` and `prod`; k8s/microk8s deferred |
| Batch model | 1 batch = N PDFs, 1 shared question; one Celery task per PDF |
| Queue delivery | Polling `GET /api/agent-runs/{id}` every ~2s |
| Async in worker | Celery task is `def` (prefork); `asyncio.run(adapter.run(...))` inside |
| Agent selector v1 | Visible dropdown; CrewAI enabled; Deep Agents/Hermes disabled with "coming soon" |
| New code location | New siblings `api/` + `web/` + `infra/`; existing dirs untouched |
| LLM selection | Config-driven YAML registry; one model per batch; local + cloud providers |
| LLM catalog storage | YAML for v1; DB catalog deferred to v2 |
| Document dedup | By `content_sha256` (same file uploaded twice returns existing row) |
| Delete behavior | Soft delete (`deleted_at`); list endpoints filter out soft-deleted |
| Result data | `transactions` stored JSONB on `agent_run_items` for v1; normalized table deferred to v2 |
| Transaction reuse | `POST /api/agent-runs` accepts pre-uploaded `document_ids[]` (dedup + reuse across batches) |
| Retry granularity | Per-item (`POST /api/agent-runs/{id}/retry?item_ids[]=...`) |
| Frontend styling | Tailwind CSS for velocity on the 4-page UI |
| FastAPI mount path | `/api` prefix (no `/v1` versioning for v1) |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                       docker-compose (infra/)                       │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │ Postgres │   │  Redis   │   │ Qdrant   │   │  FastAPI (uvicorn)│ │
│  │  :5432   │   │  :6379   │   │  :6333   │   │  :8000            │ │
│  │ SQL+     │   │ broker + │   │ vectors  │   │  async REST       │ │
│  │ Alembic  │   │ backend  │   │          │   │  SQLAlchemy 2.0   │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────────────┘ │
│                        ▲                            │              │
│                        │ enqueue                    │ enqueue      │
│                        │                            ▼              │
│  ┌─────────────────────┴────────────────────────────────────────┐ │
│  │  Celery worker (prefork)  :n/a                                 │ │
│  │  - pulls jobs from Redis                                      │ │
│  │  - asyncio.run(adapter.run(pdf, query))                        │ │
│  │  - writes Job rows back to Postgres + Qdrant                  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
│  ┌──────────────────────┐                                           │
│  │ Nginx :80            │  prod profile                             │
│  │  /    → SPA static   │  dev profile: Vite HMR :5173              │
│  │  /api → FastAPI :8000│                                           │
│  └──────────▲───────────┘                                           │
│             │ Browser                                               │
│  React + Vite SPA (polls /api/agent-runs/{id} every ~2s)            │
└─────────────────────────────────────────────────────────────────────┘
```

### Edge / port topology

- Browser → Nginx `:80` (prod) → `/api/*` proxied to FastAPI `:8000` (same origin; no CORS).
- Browser → Vite `:5173` (dev) → FastAPI `:8000` directly (no Nginx; CORS configured for `localhost:5173` in dev profile).
- Postgres `:5432`, Redis `:6379`, Qdrant `:6333` — internal-only by default. Qdrant dashboard `:6333` optionally exposed.
- Celery worker — no port.
- Vite dev server `:5173` — only in `dev` profile; not in `prod`.

### Service responsibilities

- **Postgres** — system of record. Tables: `users`, `documents`, `agent_runs`, `agent_run_items`. Alembic owns schema versions.
- **Redis** — Celery broker + result backend. No app-level caching in v1.
- **Qdrant** — production RAG store for the CrewAI adapter. Per-document collections `doc_<id>`. Same `langchain-qdrant` + `qdrant-client` deps already in root requirements.
- **FastAPI (async)** — REST surface. Accepts uploads, creates `agent_runs` + N child `agent_run_items`, enqueues one Celery task per item. Read endpoints for batch/item status, markdown report, transactions, document download. JWT login for the seeded admin.
- **Celery worker (prefork)** — one task per `agent_run_item`. Internally `asyncio.run(crewai_adapter.run(...))`. Updates item status/markdown/transactions in Postgres as it progresses. Touches Qdrant for the RAG step.
- **React + Vite** — SPA. Pages: Login, New batch (multi-file upload + shared question + agent dropdown + model dropdown), Batch list, Batch detail. Polls `GET /api/agent-runs/{id}` every ~2s while any item is pending/running.

### Request lifecycle (one batch)

```
Browser                  FastAPI                 Redis        Celery worker        Postgres      Qdrant
  │  POST /api/agent-runs    │                        │              │                  │            │
  │  (form: document_ids[N], │                        │              │                  │            │
  │   question, agent,       │                        │              │                  │            │
  │   llm_provider_id,       │                        │              │                  │            │
  │   llm_model_id)          │                        │              │                  │            │
  │ ──────────────────────►│                        │              │                  │            │
  │                        │ validate agent+provider+model            │                  │            │
  │                        │ INSERT agent_runs row  │              │                  │            │
  │                        │ INSERT N agent_run_items│             │                  │            │
  │                        │ enqueue N tasks        │              │                  │            │
  │                        │ ──────────────────────►│              │                  │            │
  │                        │ ◄─── batch_id ──────────│              │                  │            │
  │ ◄── 201 batch_id ──────│                        │              │                  │            │
  │                        │                        │ pull         │                  │            │
  │                        │                        │ ────────────►│                  │            │
  │                        │                        │              │ asyncio.run(       │           │
  │                        │                        │              │  adapter.run(...))│            │
  │                        │                        │              │ UPDATE item       │            │
  │                        │                        │              │  status=running ──┼───────────►│
  │                        │                        │              │ extract → redact  │            │
  │                        │                        │              │ → store ──────────┼───────────►│
  │                        │                        │              │ → RAG answer ◄────┼───────────►│
  │                        │                        │              │ UPDATE item       │            │
  │                        │                        │              │  status=succeeded │            │
  │                        │                        │              │  md + tx JSON     │            │
  │ GET /api/agent-runs/{id}│                       │              │                  │            │
  │ ──────────────────────►│                        │              │                  │            │
  │ ◄── batch + items ─────│                        │              │                  │            │
  │  (repeat every 2s)     │                        │              │                  │            │
```

### What stays experimental

Untouched by v1:
- `backend/app/core/ai_agent_skills_dev.ipynb` — declared a frozen reference; the adapter in `api/agents/crewai/adapter.py` ports its logic. The notebook is never imported.
- `agents/deep-agents/`, `agents/hermes/`, `agents/crewai/` — experiment harnesses stay as-is.
- `frontend/streamlit_app/` — old Streamlit prototypes remain for reference.
- `notebooks/`, `yolo-base-layout-analysis/` — earlier exploratory prototypes untouched.

`backend/app/skills/` (the five domain `SKILL.md` packages) is imported as-is by the CrewAI adapter via `PYTHONPATH=/app/backend` in the API image.

## Data model

Five tables, SQLAlchemy 2.0 typed `Mapped` style. All have `created_at`/`updated_at`/`deleted_at` timestamptz unless noted.

### `users` — one seeded admin row

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `email` | str unique not null | |
| `password_hash` | str not null | bcrypt via `passlib` |
| `is_admin` | bool default true | |
| `created_at`, `updated_at` | timestamptz | (no `deleted_at` for v1) |

### `documents` — uploaded PDF metadata, deduplicated by hash

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `owner_id` | UUID FK `users.id` not null | |
| `original_filename` | str not null | |
| `storage_path` | str not null | under `data/uploads/<yyyy>/<mm>/<doc_id>.pdf` |
| `content_sha256` | str unique not null | dedupe key; same file uploaded twice returns the same row |
| `mime_type` | str default `'application/pdf'` | |
| `size_bytes` | bigint | |
| `page_count` | int null | populated on first parse |
| `qdrant_collection` | str null | `doc_<id>`; set after first successful RAG store |
| `created_at`, `deleted_at` | timestamptz | soft delete |

### `agent_runs` — batch parent (1 batch = N PDFs + 1 shared question)

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `owner_id` | UUID FK `users.id` not null | |
| `agent` | str not null | `'crewai' \| 'deep-agents' \| 'hermes'`; DB-level CHECK |
| `question` | text not null | |
| `status` | str not null default `'pending'` | CHECK in `{pending, running, succeeded, partial, failed}` |
| `llm_provider` | str not null | short tag, e.g. `lm-studio`, `openrouter`, `gemini` |
| `llm_model` | str not null | LiteLLM model string, e.g. `openai/qwen2.5-14b-instruct` |
| `started_at`, `finished_at` | timestamptz null | |
| `created_at`, `deleted_at` | timestamptz | soft delete |

### `agent_run_items` — per-PDF child job (one Celery task each)

| Column | Type | Notes |
|---|---|---|
| `id` | UUID PK | |
| `run_id` | UUID FK `agent_runs.id` ondelete cascade not null | |
| `document_id` | UUID FK `documents.id` not null | |
| `celery_task_id` | str null | |
| `status` | str not null default `'pending'` | CHECK in `{pending, running, succeeded, failed}` |
| `error` | text null | |
| `markdown_report` | text null | agent's final Markdown |
| `transactions` | jsonb null | array of `{date, description, credit, debit, balance, currency}` |
| `started_at`, `finished_at` | timestamptz null | |
| index | | `(run_id, status)`; `(document_id)` |

### `alembic_version` — managed by Alembic

### Status machine

`agent_run_items.status ∈ {pending, running, succeeded, failed}`

`agent_runs.status ∈ {pending, running, succeeded, partial, failed}`

Parent status derived from children on each item transition:
- all pending → `pending`
- any running / mixed pending + succeeded → `running`
- all succeeded → `succeeded`
- some succeeded + some failed → `partial`
- all failed → `failed`

### Design notes

- Two-table batch model allows partial success — some PDFs fail, the rest still render in the UI.
- `transactions` as JSONB for v1 keeps the Alembic migration surface small; normalized table deferred to v2 if we need per-transaction queries or charts.
- Document dedup via `content_sha256` saves re-chunking/re-embedding when the same statement is re-uploaded (common in personal-finance use).
- All docs belong to the seeded admin; `owner_id` FK is forward-compatible with v2 multi-user.

### Seed behaviour

First migration (`0001_initial_schema`) inserts the admin row using `ADMIN_EMAIL` and `ADMIN_PASSWORD_HASH` env vars (bcrypt hash precomputed; plaintext password never in env). The startup refuses to boot if either env var is absent — forces the operator to seed a hash, never a default password.

## LLM provider & model selection

### Schema columns on `agent_runs`

- `llm_provider: str not null` — short tag: `lm-studio` | `openrouter` | `gemini` | `openai` | `ollama` | `groq` ...
- `llm_model: str not null` — the LiteLLM model string used, e.g. `openai/qwen2.5-14b-instruct`, `openrouter/google/gemini-2.5-flash`, `gemini/gemini-2.5-flash`.

No catalog table for v1 — registry is config-driven (see below). A DB catalog + admin CRUD UI is v2 work.

### `LLMProviderRegistry`

A Pydantic-validated config file `api/config/llm_providers.yaml` (loaded at startup, hot-reloadable in dev) defines the catalog. Each entry:

```yaml
providers:
  - id: lm-studio
    display_name: "LM Studio (local)"
    kind: local
    base_url: http://lm-studio:1234/v1     # compose service or host.docker.internal
    api_key_env: LM_STUDIO_API_KEY         # read from env; absent = "lm-studio" sentinel OK
    models:
      - id: openai/qwen2.5-14b-instruct
        display_name: "Qwen2.5 14B Instruct"
      - id: openai/google/gemma-3-12b-qat
        display_name: "Gemma 3 12B QAT"

  - id: openrouter
    display_name: "OpenRouter (cloud)"
    kind: cloud
    base_url: https://openrouter.ai/api/v1
    api_key_env: OPENROUTER_API_KEY
    models:
      - id: openrouter/google/gemini-2.5-flash
        display_name: "Gemini 2.5 Flash (via OpenRouter)"
      - id: openrouter/anthropic/claude-3.5-sonnet
        display_name: "Claude 3.5 Sonnet"

  - id: gemini
    display_name: "Google Gemini (direct)"
    kind: cloud
    api_key_env: GOOGLE_API_KEY
    models:
      - id: gemini/gemini-2.5-flash
        display_name: "Gemini 2.5 Flash"

  - id: ollama
    display_name: "Ollama (local)"
    kind: local
    base_url: http://ollama:11434
    models:
      - id: ollama/llama3.2
        display_name: "Llama 3.2"
```

### Runtime resolution

The CrewAI adapter receives `(llm_provider_id, llm_model_id)`, looks up the provider, and assembles the LiteLLM call with the resolved `base_url` (if any) and `api_key` (read from env; never echoed back). For direct providers (Gemini, openai) there is no `base_url`; LiteLLM routes by the model prefix.

The CrewAI `LLM(model=..., base_url=..., api_key=...)` factory is used for the Crew path; direct LiteLLM `acompletion` is used for the structured-extraction path (instructor + pydantic `Transaction` models). Both are already used in the notebook; the adapter parameterizes them instead of hard-coding.

### Availability probing

FastAPI startup and `GET /api/llm-models` run a cheap reachability check per provider:
- `kind: local` — TCP/HTTP ping to `base_url`.
- `kind: cloud` — verifies the corresponding `*_API_KEY` env var is set.

Each provider in the `/llm-models` response carries `available: bool` and `unavailable_reason: str | null`. The result is cached server-side for ~30s to avoid hammering on every frontend refresh.

### Frontend model selector

On the "New batch" page, above the question field:

```
Agent:   [ CrewAI ▾ ]
Model:   [ Local · LM Studio · Qwen2.5 14B Instruct                 ▾ ]
           (grouped: "Local — LM Studio (3 models)", "Cloud — OpenRouter (12)", "Cloud — Gemini direct (2)")
           (groups with available:false show a lock icon + tooltip "API key not set" / "LM Studio not reachable")
```

- Dropdown populated from `GET /api/llm-models` on page mount; cached in TanStack Query with 60s refetch.
- Selection persisted in `localStorage` as last-used default.
- Only `available:true` models are selectable; unavailable ones remain visible but disabled with a tooltip explaining why.
- Selected `llm_provider_id` + `llm_model_id` sent with `POST /api/agent-runs`.
- Batch detail page shows the model used as a read-only badge.

### Why config-driven for v1

- Provider/model lists change slowly and don't need per-user editing yet (single admin user).
- YAML is reviewable in git and easy to extend without a migration.
- v2 can promote to DB tables (`llm_providers` + `llm_models`) + admin CRUD UI without changing the API contract.

## Directory layout (new code)

Three new top-level directories added alongside existing `backend/`, `agents/`, `frontend/`, `notebooks/`, `yolo-base-layout-analysis/` (all untouched).

```
repo/
├── api/                                 # NEW — FastAPI service + worker
│   ├── pyproject.toml                   # uv/pip project (Python 3.11)
│   ├── requirements.txt                 # pinned; mirrors backend pin style
│   ├── alembic.ini
│   ├── README.md
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                       # FastAPI app factory, middleware, lifespan
│   │   ├── config.py                     # pydantic-settings: DB/Redis/Qdrant/seed admin
│   │   ├── deps.py                       # FastAPI deps: DB session, current_user, registry
│   │   ├── db/
│   │   │   ├── __init__.py
│   │   │   ├── session.py                 # async engine + sessionmaker
│   │   │   └── base.py                    # SQLAlchemy DeclarativeBase
│   │   ├── models/                        # SQLAlchemy ORM (typed Mapped)
│   │   │   ├── __init__.py
│   │   │   ├── user.py
│   │   │   ├── document.py
│   │   │   ├── agent_run.py              # AgentRun + AgentRunItem
│   │   │   └── mixins.py                # TimestampMixin
│   │   ├── schemas/                      # Pydantic v2 request/response
│   │   │   ├── auth.py
│   │   │   ├── llm.py                    # LLM provider/model schemas
│   │   │   ├── document.py
│   │   │   └── agent_run.py
│   │   ├── api/                           # routers
│   │   │   ├── __init__.py
│   │   │   ├── auth.py                    # /auth/login, /auth/me
│   │   │   ├── llm.py                     # /llm-models
│   │   │   ├── documents.py              # CRUD + download
│   │   │   └── agent_runs.py             # list, create, detail, retry
│   │   ├── core/
│   │   │   ├── security.py                # JWT + bcrypt (python-jose, passlib)
│   │   │   ├── storage.py                 # PDF disk storage under data/uploads/
│   │   │   └── hashing.py                 # sha256 for doc dedup
│   │   ├── agents/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                    # BaseAgentAdapter ABC + AgentResult pydantic
│   │   │   ├── registry.py               # AgentRegistry: id->adapter, enabled flags
│   │   │   ├── llm_provider.py            # LLMProviderRegistry + load llm_providers.yaml
│   │   │   └── crewai/
│   │   │       ├── __init__.py
│   │   │       ├── adapter.py             # CrewAIAdapter(BaseAgentAdapter)
│   │   │       ├── extractor.py          # pydantic + litellm structured tx extraction
│   │   │       └── rag.py                 # Qdrant collection mgmt + query
│   │   ├── worker/
│   │   │   ├── __init__.py
│   │   │   ├── celery_app.py             # Celery broker=redis, prefork pool
│   │   │   ├── tasks.py                  # @agent_task(merge:agent_run_item_id)
│   │   │   └── runner.py                  # asyncio.run(adapter.run(...))
│   │   └── tests/
│   │       ├── conftest.py                # pytest fixtures: db, client, seeded admin
│   │       ├── test_api_auth.py
│   │       ├── test_api_llm.py
│   │       ├── test_api_documents.py
│   │       ├── test_api_agent_runs.py
│   │       ├── test_registry.py
│   │       ├── test_worker_tasks.py       # uses eager Celery in test
│   │       └── test_crewai_adapter.py     # mocked LLM + Qdrant
│   ├── alembic/
│   │   ├── env.py
│   │   ├── script.py.mako
│   │   └── versions/
│   │       └── 0001_initial_schema.py     # users, documents, agent_runs, items, seed admin
│   └── config/
│       └── llm_providers.yaml             # LLM provider/model catalog (reviewable)
│
├── web/                                  # NEW — React + Vite
│   ├── package.json
│   ├── vite.config.ts
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── eslint.config.js
│   ├── index.html
│   ├── nginx.conf                        # SPA fallback + /api proxy (prod)
│   ├── Dockerfile                        # multi-stage: build → nginx:alpine
│   └── src/
│       ├── main.tsx
│       ├── App.tsx
│       ├── router.tsx
│       ├── api/
│       │   ├── client.ts                 # fetch wrapper + 401 redirect
│       │   ├── auth.ts
│       │   ├── llm.ts
│       │   ├── documents.ts
│       │   └── agentRuns.ts
│       ├── components/
│       │   ├── Layout.tsx
│       │   ├── ProtectedRoute.tsx
│       │   ├── AgentDropdown.tsx
│       │   ├── ModelDropdown.tsx
│       │   ├── PdfMultiDropzone.tsx
│       │   ├── BatchItemCard.tsx
│       │   ├── MarkdownViewer.tsx
│       │   └── TransactionsTable.tsx
│       ├── pages/
│       │   ├── LoginPage.tsx
│       │   ├── NewBatchPage.tsx
│       │   ├── BatchListPage.tsx
│       │   └── BatchDetailPage.tsx
│       ├── hooks/
│       │   ├── useAuth.ts
│       │   ├── useBatchPolling.ts        # 2s poll while pending/running
│       │   └── useLLMModels.ts          # react-query, 60s refresh
│       ├── stores/
│       │   └── authStore.ts              # zustand: token + user
│       ├── types/                        # TS interfaces mirroring API schemas
│       ├── utils/
│       └── styles/
│
├── infra/                               # NEW — deployment manifests
│   ├── docker-compose.yml               # profiles: dev, prod; all services
│   ├── docker-compose.override.yml      # dev-only: FastAPI reload, Vite HMR, source mounts
│   ├── .env.example                     # documented keys; secrets excluded
│   ├── api.Dockerfile                   # FastAPI + worker same image, entrypoint-driven
│   ├── postgres-init.sql                # optional DB extensions (e.g. pgcrypto), idempotent
│   ├── qdrant-init.yaml                 # optional initial collections (empty for v1)
│   └── k8s/                            # v2 placeholder
│       └── README.md                    # "K8s/microk8s manifests deferred to post-v1"
│
├── backend/                             # UNCHANGED — CrewAI notebook + skills (frozen ref)
├── agents/                              # UNCHANGED — experiment harnesses
├── frontend/streamlit_app/              # UNCHANGED — old Streamlit prototypes
├── notebooks/                          # UNCHANGED
├── yolo-base-layout-analysis/          # UNCHANGED
├── data/                               # SHARED — uploads/ now also written here by api
│   ├── bank-statement-document/        # existing samples
│   ├── uploads/<yyyy>/<mm>/<doc_id>.pdf # NEW: API stores uploads here
│   └── vector_stores/                  # legacy; Qdrant lives in compose service
└── docs/
    └── superpowers/specs/
        └── 2026-07-25-full-stack-app-design.md   # this spec
```

### Calls out

1. **One Docker image for API + Worker** (`infra/api.Dockerfile`) — same image, different entrypoint (`uvicorn api.app.main:app` vs `celery -A api.app.worker.celery_app worker`). Keeps deps in lockstep; matches the "thin wrapper" scope.
2. **Importing existing skills**: `api/` Dockerfile sets `PYTHONPATH=/app/backend` so `from backend.app.skills.crewai_skills_loader import load_skills` works. The notebook itself is never imported; `api/agents/crewai/adapter.py` ports its logic.
3. **Web stack**: React 18 + TypeScript strict + Vite + React Router + TanStack Query + zustand + Tailwind. `nginx.conf` does SPA fallback + `/api` → `http://api:8000`.
4. **Shared `data/`**: API writes uploads under `data/uploads/`; the compose volume mount is shared. Existing `data/bank-statement-document/` samples remain readable so the adapter supports an optional "use existing sample" path without re-upload.
5. **`infra/k8s/`** is a stub per the user's "compose first, k8s/microk8s later" decision; README notes it's deferred.

## API surface (OpenAPI contract)

All endpoints under `/api` (FastAPI mounted at `/api`; OpenAPI at `/api/openapi.json`; Swagger at `/api/docs`). JWT bearer auth required everywhere except `/api/auth/login` and `/api/health`.

### Auth

| Method | Path | Body / params | Response | Notes |
|---|---|---|---|---|
| POST | `/api/auth/login` | `{email, password}` | `{access_token, token_type:"bearer", user:{id,email}}` | bcrypt verify; 15-min access TTL |
| GET | `/api/auth/me` | — | `{id,email,is_admin}` | JWT required |

### LLM catalog

| Method | Path | Response | Notes |
|---|---|---|---|
| GET | `/api/llm-models` | `{providers:[{id, display_name, kind:"local"\|"cloud", available, unavailable_reason, models:[{id, display_name}]}]}` | Probes availability per provider; 30s server-side cache |

### Documents

| Method | Path | Body / params | Response | Notes |
|---|---|---|---|---|
| POST | `/api/documents` | multipart `file` | `{id, original_filename, size_bytes, content_sha256, deduplicated}` | Dedup by sha256; existing row returned with `deduplicated:true` |
| GET | `/api/documents` | `?limit=50&offset=0` | `{items:[...], total}` | Paginated list of owner's docs; filters `deleted_at IS NULL` |
| GET | `/api/documents/{id}` | — | `{id, original_filename, mime_type, size_bytes, page_count, created_at}` | Metadata only |
| GET | `/api/documents/{id}/content` | — | `application/pdf` stream | Inline download; Content-Disposition |
| DELETE | `/api/documents/{id}` | — | `204` | Soft delete only (sets `deleted_at`); refuses if referenced by an item with `status ∈ {pending, running}` |

### Agent runs (batches)

| Method | Path | Body / params | Response | Notes |
|---|---|---|---|---|
| POST | `/api/agent-runs` | JSON body: `{document_ids:[UUID], agent?, question, llm_provider_id, llm_model_id}` | `201 {id, status:"pending", items:[{id, document_id, status:"pending"}]}` | Validates agent/provider/model via registries; creates 1 run + N items; enqueues N Celery tasks. Files are pre-uploaded via `/api/documents` so no multipart here. |
| GET | `/api/agent-runs` | `?limit=20&offset=0&status=` | `{items:[...], total}` | Owner's batches, newest first; filters `deleted_at IS NULL` |
| GET | `/api/agent-runs/{id}` | — | `{id, agent, question, llm_provider, llm_model, status, created_at, started_at, finished_at, items:[{id, document_id, status, error, markdown_report, transactions, started_at, finished_at}]}` | Main polling endpoint; 200 even if batch failed |
| POST | `/api/agent-runs/{id}/retry` | query `?item_ids[]=...` optional | `202 {retried_item_ids:[...]}` | Re-enqueues items. Default = all `status='failed'` items; if `item_ids[]` provided, must all be `failed` (others → 409 CONFLICT). Same run + question + llm provider/model. New Celery task ids written to `celery_task_id`. |
| DELETE | `/api/agent-runs/{id}` | — | `204` | Cancels pending Celery tasks (revoke); marks items/run soft-deleted; keeps docs |

### Polling contract

- `GET /api/agent-runs/{id}` returns `200` even when the run failed — the `status` field is the source of truth. Items carry their own `status`, `error` (string), `markdown_report`, and `transactions` (JSONB array).
- Frontend polls every ~2s while `run.status ∈ {pending, running}`; stops on `{succeeded, partial, failed}` and renders results. A soft-deleted run (DELETE endpoint) disappears from the list via `deleted_at IS NULL` filtering — no `cancelled` state needed.

### Error envelope (uniform)

```json
{ "error": { "code": "VALIDATION_ERROR", "message": "...", "details": {...} } }
```

Codes:
- `VALIDATION_ERROR` (422) — request body / form validation
- `UNAUTHORIZED` (401) — missing/invalid token
- `FORBIDDEN` (403) — token valid but not the owner
- `NOT_FOUND` (404)
- `CONFLICT` (409) — e.g. retry on a running item
- `PROVIDER_UNAVAILABLE` (422) — selected LLM provider/model not enabled or unreachable
- `INTERNAL` (500) — unhandled; logged

FastAPI exception handlers register these and never leak tracebacks to the client.

### Health

- `GET /api/health` — `{status:"ok", db:"ok"|"down", redis:"ok"|"down", qdrant:"ok"|"down"}` (no auth; used by compose healthchecks / readiness probes).

## CrewAI adapter design

### `BaseAgentAdapter` contract (`api/agents/base.py`)

```python
class BaseAgentAdapter(ABC):
    name: str                       # 'crewai', 'deep-agents', 'hermes'
    display_name: str
    enabled: bool                   # False for Deep Agents/Hermes in v1
    description: str

    @abstractmethod
    async def run(
        self,
        *,
        pdf_path: str,
        question: str,
        llm_provider_id: str,
        llm_model_id: str,
        agent_run_item_id: UUID,
    ) -> AgentResult: ...
```

```python
class AgentResult(BaseModel):
    markdown_report: str
    transactions: list[Transaction]   # pydantic, same shape as notebook cell 67
    raw: dict | None = None           # provider-specific extras (e.g. MLflow run id)
```

### `CrewAIAdapter` (`api/agents/crewai/adapter.py`)

Ports the notebook's `analyze_bank_statement(pdf_path, query)` flow into a single module without notebook globals:

1. **Load skills** — `from backend.app.skills.crewai_skills_loader import load_skills` (PYTHONPATH=/app/backend in the image); activates the 5 `SKILL.md` packages.
2. **Build agents + tasks + crew** — same definitions as notebook cells 25–77 (`bank_statement_agent`, `load_document_task`, `store_task`, `Financial_Analytic_task`...). Crew `Process.sequential`.
3. **LLM** — `crewai.LLM(model=llm_model_id, base_url=resolved_base_url, api_key=resolved_key)` for the Crew path. `litellm.acompletion(model=..., base_url=..., api_key=..., messages=...)` for the structured-extraction path (instructor + pydantic `Transaction`).
4. **Vector store** — `langchain_qdrant.QdrantVectorStore` pointed at the compose Qdrant service (`QDRANT_URL`); per-document collection `doc_<document_id>`. Reuse if collection exists; otherwise chunk + embed + upsert. Embedding model resolved from env (`EMBEDDING_MODEL` default `intfloat/multilingual-e5-small` dim 768).
5. **Pii redaction** — applied before vector store, exactly as the `pii-handling` skill mandates (regex-based for v1; NER deferred).
6. **Structured tx extraction** — instructor + pydantic parses the answer into `list[Transaction]` JSONB for the `agent_run_items.transactions` column.
7. **Kick off** — `await crew.kickoff_async(inputs={pdf_path, query})`; wrap in try/except; on failure, raise so the Celery task marks the item `failed` with the error string.
8. **Observability** — MLflow autolog enabled in dev profile only; disabled in prod to avoid the `backend/app/core/mlflow.db`-style sprawl in the production container.

### `AgentRegistry` (`api/agents/registry.py`)

```python
class AgentRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, BaseAgentAdapter] = {}

    def register(self, adapter: BaseAgentAdapter) -> None: ...
    def get(self, name: str) -> BaseAgentAdapter: ...    # 404 if unknown
    def list(self) -> list[AgentInfo]: ...              # name, display_name, enabled, description
```

v1 registers `CrewAIAdapter` (enabled=True) + `DeepAgentsAdapter` stub (enabled=False) + `HermesAdapter` stub (enabled=False). Selecting a disabled adapter returns `PROVIDER_UNAVAILABLE` 422.

### Why one-time port over subprocess/notebook import

- `.ipynb` can't be imported; `nbconvert --execute` is slow and stateful.
- A clean adapter gives a typed interface Deep Agents and Hermes will mirror later.
- Structured pydantic transaction extraction needs Python objects, not stdout parsing.
- The notebook is frozen as a reference; the adapter is now authoritative. A pytest suite fixes the contract.

## Worker design

### `celery_app.py`

```python
celery_app = Celery("api", broker=settings.REDIS_URL, backend=settings.REDIS_URL)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,            # one task at a time per process
    task_time_limit=60 * 10,                 # 10 min hard
    task_soft_time_limit=60 * 8,             # 8 min soft → marks item failed cleanly
    timezone="UTC",
)
```

Prefork pool (default). `worker_prefetch_multiplier=1` prevents a single worker from hoarding tasks while another is idle.

### `tasks.py`

```python
@celery_app.task(name="agent.run_item", bind=True, max_retries=2)
def run_agent_item(self, agent_run_item_id: str) -> None:
    asyncio.run(_runner(agent_run_item_id))
```

`asyncio.run` creates a fresh event loop per task; no loop sharing across prefork processes.

### `runner.py`

```python
async def _runner(item_id: UUID) -> None:
    item = await load_item(item_id)
    await mark_item(item_id, status="running")
    try:
        adapter = registry.get(item.run.agent)
        result = await adapter.run(
            pdf_path=item.document.storage_path,
            question=item.run.question,
            llm_provider_id=item.run.llm_provider,
            llm_model_id=item.run.llm_model,
            agent_run_item_id=item.id,
        )
        await mark_item_succeeded(item_id, markdown=result.markdown_report, tx=result.transactions)
    except Exception as exc:
        await mark_item_failed(item_id, error=str(exc))
        raise self.retry(exc=exc, countdown=2 ** self.request.retries)
    finally:
        await refresh_run_status(item.run_id)
```

- `refresh_run_status` recomputes parent `agent_runs.status` from children (per the status machine).
- Failure sets the item `failed` with the error string; the Celery `max_retries=2` retries the task with exponential backoff before settling on `failed`.
- `agent_run_items.error` truncated to ~4 KB to avoid huge tracebacks in Postgres.

## Frontend design

### Pages

| Page | Route | Purpose |
|---|---|---|
| `LoginPage` | `/login` | Email + password; stores JWT in `localStorage` via `authStore` |
| `NewBatchPage` | `/batches/new` | Multi-PDF dropzone, agent dropdown, model dropdown (grouped Local/Cloud with availability), shared question textarea; submits via `POST /api/documents` per file then `POST /api/agent-runs`; on success redirects to `BatchDetailPage` |
| `BatchListPage` | `/batches` | Paginated list of own batches; columns: created, agent, model, status, item counts; click → detail |
| `BatchDetailPage` | `/batches/:id` | Per-item cards with status badge; succeeded cards expand to Markdown viewer + Transactions table; failed cards show error + retry button; `useBatchPolling` runs every 2s while `status ∈ {pending, running}` |

### Components

- `Layout` — top bar with app name, current user, logout; Outlet.
- `ProtectedRoute` — redirects to `/login` if no token.
- `AgentDropdown` — fetches agents (static list for v1; CrewAI enabled, others locked).
- `ModelDropdown` — fetches `GET /api/llm-models`; groups by `kind`; disabled entries show lock + tooltip with `unavailable_reason`.
- `PdfMultiDropzone` — drag-and-drop multi-file; uploads to `/api/documents` immediately; shows dedup badge (`deduplicated:true`).
- `BatchItemCard` — one per `agent_run_item`; status pill; spinner when running; expandable on success.
- `MarkdownViewer` — renders `markdown_report` (use `react-markdown` + `remark-gfm`).
- `TransactionsTable` — renders `transactions` JSONB; sortable columns.

### Stack

- React 18 + TypeScript strict
- Vite 5
- React Router 6
- TanStack Query 5 (server state; 60s refetch on `/llm-models`)
- zustand (auth store)
- Tailwind CSS
- `react-markdown` + `remark-gfm`
- Vitest + React Testing Library for component tests

### Polling

`useBatchPolling(batchId)`:
- 2s interval
- stops when `run.status ∈ {succeeded, partial, failed}`
- on stop, refetches one last time to get final results
- cleans up the interval on unmount

## Deployment (Docker compose)

Two profiles in `infra/docker-compose.yml`:

### Shared services (both profiles)

- `postgres:16` — volume `pgdata`; `postgres-init.sql` enables `pgcrypto` for `gen_random_uuid()`
- `redis:7-alpine` — no persistence needed (broker)
- `qdrant/qdrant:v1.x` — volume `qdrant_data`; `:6333` internal
- `worker` — image `api:dev`; entrypoint `celery -A api.app.worker.celery_app worker --loglevel=info`; depends_on postgres+redis+qdrant

### `dev` profile

- `api` — image built from `infra/api.Dockerfile`; mounted source `./api:/app/api` and `./backend:/app/backend` and `./data:/app/data`; `uvicorn api.app.main:app --reload --host 0.0.0.0 --port 8000`; `PYTHONPATH=/app:/app/backend`
- `web` — `node:20-alpine`; mounted `./web:/app`; `npm run dev -- --host 0.0.0.0`; `:5173` exposed; Vite proxy `/api → http://api:8000`; CORS on FastAPI for `localhost:5173`

### `prod` profile

- `api` — built image; `uvicorn api.app.main:app --host 0.0.0.0 --port 8000`; no source mount; `PYTHONPATH=/app:/app/backend`
- `web` — multi-stage `web/Dockerfile` (`node:20` build → `nginx:alpine` serve); `nginx.conf` serves built SPA on `:80` and proxies `/api → http://api:8000`; same origin; no CORS

### Required env vars (`infra/.env.example`)

```
# Postgres
POSTGRES_USER=bankai
POSTGRES_PASSWORD=change_me
POSTGRES_DB=bankai

# Redis
REDIS_URL=redis://redis:6379/0

# Qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=

# Auth
JWT_SECRET=change_me_to_a_long_random_string
JWT_ALG=HS256
JWT_TTL_MINUTES=15
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD_HASH=$2b$12$...bcrypt_hash...

# Embeddings
EMBEDDING_MODEL=intfloat/multilingual-e5-small

# LLM provider keys (consumed by LLMProviderRegistry via api_key_env)
LM_STUDIO_API_KEY=lm-studio
OPENROUTER_API_KEY=
GOOGLE_API_KEY=
OPENAI_API_KEY=
GROQ_API_KEY=
```

`infra/.env.example` committed; real `.env` is gitignored. `.gitignore` updated to exclude `infra/.env`, `api/app/core/*.db` (shouldn't exist in production but defensive), `web/node_modules`, `web/dist`, `data/uploads/`.

## Testing strategy

| Layer | Tool | What | Example |
|---|---|---|---|
| Unit (agents) | pytest + `pytest-asyncio` | Adapter with mocked LiteLLM (`litellm.acompletion` monkeypatch) + fake Qdrant client | `test_crewai_adapter_returns_markdown_and_tx()` |
| Unit (worker) | pytest + eager Celery (`task_always_eager=True`) | Task updates DB rows correctly on success/failure | `test_task_marks_item_succeeded_and_writes_markdown()` |
| API integration | pytest + `httpx.AsyncClient` + async SQLAlchemy + sqlite-in-memory or test Postgres | Auth, documents CRUD/dedup, agent-run create→enqueue→poll, retry, soft delete | `test_create_run_enqueues_n_tasks()` |
| Registry | pytest | AgentRegistry + LLMProviderRegistry validation (unknown agent/provider/model rejected) | `test_unknown_provider_rejected()` |
| LLM availability | pytest with mocked HTTP probes | `kind:local` ping failure → `available:false` + reason | `test_local_provider_unavailable_when_ping_fails()` |
| Frontend | Vitest + React Testing Library | Component tests for dropdowns, batch card states, polling hook | `test_useBatchPolling_stops_on_terminal()` |
| E2E smoke | manual / deferred to v2 | Full upload→run→view with real LM Studio | (documented; not automated in v1) |

### Conventions (matched to existing `agents/deep-agents/tests/`)

- pytest fixtures in `conftest.py`; `tmp_path` for disk storage.
- Deterministic test data: existing `data/bank-statement-document/Dummy-Bank-Statement.pdf` fixture.
- No real LLM/Qdrant in CI; everything mocked.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| CrewAI notebook port drift | `backend/app/core/ai_agent_skills_dev.ipynb` declared frozen reference; adapter is authoritative; pytest suite on the adapter |
| Local model availability varies by host | `LLMProviderRegistry` probes at startup + on `/llm-models`; UI disables unavailable |
| Qdrant collection sprawl (per-doc) | `doc_<id>` namespacing; re-run reuses; v2 retention policy |
| Long-running PDF parse blocks worker | One Celery task per item; `task_time_limit=600` / `task_soft_time_limit=480`; item marked `failed` on timeout |
| Secrets (LM Studio sentinel, cloud keys) | `.env` via pydantic-settings; `.env.example`; never echo keys; `api_key_env` indirection in YAML |
| Mixed pool: asyncio in prefork worker | `asyncio.run()` per task (no event loop sharing); tested with eager mode |
| YAML catalog drift vs real providers | YAML is catalog of record; v2 promotes to DB if drift becomes a problem |
| Mass PII in Qdrant | `pii-handling` skill + redact tool applied before store; workspace not committed; Qdrant volume is host-local only |
| Cold-start delay on first LM Studio call | UI surfaces `running` status; soft time limit generous (8 min); user retries at item level |
| Bootstrap without `ADMIN_PASSWORD_HASH` | FastAPI lifespan refuses to start, prints the bcrypt command; no silent default |

## v1 acceptance criteria

1. `docker-compose --profile dev up` brings up Postgres, Redis, Qdrant, FastAPI, Vite, and Worker; `GET /api/health` returns all-green.
2. Alembic migration `0001_initial_schema` creates `users`, `documents`, `agent_runs`, `agent_run_items`; seeds one admin from env.
3. `POST /api/auth/login` returns JWT; `GET /api/auth/me` echoes the admin.
4. `GET /api/llm-models` returns the YAML catalog with live `available` flags.
5. `POST /api/documents` uploads a PDF, dedups by sha256, stores under `data/uploads/`.
6. `POST /api/agent-runs` with N `document_ids` + question + agent + provider + model creates a run + N items and enqueues N Celery tasks.
7. Worker runs each item: extract → redact → Qdrant store → RAG answer → write markdown + transactions JSONB; item status transitions pending→running→succeeded.
8. `GET /api/agent-runs/{id}` reflects live status; frontend polls and renders Markdown + transactions table.
9. `POST /api/agent-runs/{id}/retry` re-enqueues failed items with the same question + model.
10. Soft delete on `agent_runs` and `documents` (`deleted_at`); list filters exclude soft-deleted.
11. Agent dropdown shows CrewAI enabled; Deep Agents + Hermes disabled with "coming soon".
12. Model dropdown shows grouped local/cloud providers, disables unavailable ones with tooltips.
13. Nginx prod profile serves built SPA on `:80` and proxies `/api/*` to FastAPI.
14. pytest suite passes (unit + API integration); frontend Vitest passes.
15. `.env.example` documents all required keys; no secrets committed; `.gitignore` excludes `data/uploads/`, `infra/.env`.

## Future work (post v1)

- Deep Agents adapter wiring (subprocess runner reusing `agents/deep-agents/run_e2e.py` per item).
- Hermes adapter wiring (Docker-sandboxed `hermes chat --config ... -q "..."` per item; security model already documented).
- Finance dashboard v2: charts (totals, credits/debits over time, spending categories) over stored `transactions`.
- Forecasting / anomaly detection on stored transactions (the "Prediction" in the project name).
- DB-backed LLM catalog + admin CRUD UI; promote YAML to tables.
- Streaming progress via SSE/WebSocket (replaces polling).
- Multi-user accounts + signup + per-user workspaces.
- Normalized `transactions` table replacing JSONB when per-row queries/charts are needed.
- Kubernetes manifests (microk8s) under `infra/k8s/`.
- Stronger PII (NER for names/addresses) and production embeddings for the Deep Agents path.

## References

- Existing crewai baseline: `backend/app/core/ai_agent_skills_dev.ipynb`
- Existing skills loader: `backend/app/skills/crewai_skills_loader.py:7`
- Deep Agents E2E (reference for symmetric adapter): `agents/deep-agents/agent.py:58`, `agents/deep-agents/run_e2e.py:11`
- Hermes E2E (reference for symmetric adapter): `agents/hermes/scripts/run_e2e.sh:38`
- LiteLLM usage: `backend/app/core/ai_agent_skills_dev.ipynb` cell 62; `README.md:170-193`
- Qdrant usage: `langchain-qdrant`, `qdrant-client` in `requirements.txt`
- Multi-harness design (parent): `docs/superpowers/specs/2026-07-20-multi-harness-agents-design.md`
- Multi-harness guide: `docs/guides/multi-harness-agents-guide.md`
- README roadmap: `README.md:211-219`