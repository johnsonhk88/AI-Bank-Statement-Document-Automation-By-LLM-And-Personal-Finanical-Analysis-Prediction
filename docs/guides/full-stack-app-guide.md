# Full-Stack App — Step-by-Step Run & Test Guide

This guide walks through **setup, running, and testing** the full-stack v1 application under `api/` + `web/` + `infra/`.

| Component | Path | Tech | Port |
|-----------|------|------|------|
| **API** | `api/` | FastAPI async + SQLAlchemy 2.0 + Alembic | `8000` (internal; proxied via nginx in prod) |
| **Worker** | `api/app/worker/` | Celery (prefork) | — (no port) |
| **Frontend** | `web/` | React 18 + Vite 5 + Tailwind | `5173` (dev only) |
| **Nginx** | `web/nginx.conf` | Serves built SPA + proxies `/api/*` | `80` (prod only) |
| **Postgres** | compose | 16-alpine | `5432` |
| **Redis** | compose | 7-alpine | `6379` |
| **Qdrant** | compose | v1.13.4 | `6333` |

**Design:** [docs/superpowers/specs/2026-07-25-full-stack-app-design.md](../superpowers/specs/2026-07-25-full-stack-app-design.md)
**Plan:** [docs/superpowers/plans/2026-07-25-full-stack-app.md](../superpowers/plans/2026-07-25-full-stack-app.md)

---

## 1. Prerequisites

### Required

- **Docker Engine** + **Docker Compose v2** (`docker compose` subcommand, not `docker-compose`)
- **Node.js 20+** (for running the frontend dev server locally; optional if using prod profile only)
- **Python 3.11+** (for running backend tests on the host; optional if testing inside the container)
- Sample PDF: `data/bank-statement-document/Dummy-Bank-Statement.pdf` (already in the repo; used for smoke testing)

### Optional — LLM provider

The app uses **LiteLLM** under the hood. Choose one of:

| Provider | What you need | Notes |
|----------|---------------|-------|
| **LM Studio** (local, free) | [LM Studio](https://lmstudio.ai/) running on `:1234` with a model loaded | Default in `llm_providers.yaml`; macOS/Windows use `host.docker.internal`; Linux see troubleshooting section |
| **Ollama** (local, free) | [Ollama](https://ollama.com/) running with e.g. `ollama pull llama3.2` | |
| **OpenRouter** (cloud, paid) | `OPENROUTER_API_KEY` from [openrouter.ai](https://openrouter.ai/) | |
| **Google Gemini** (cloud, free tier) | `GOOGLE_API_KEY` from [aistudio.google.com](https://aistudio.google.com/) | |
| **OpenAI** (cloud, paid) | `OPENAI_API_KEY` from [platform.openai.com](https://platform.openai.com/) | |

For **CrewAI** to work, you must have at least one provider configured and reachable.

---

## 2. Configuration

```bash
# 1. Copy the environment template
cp infra/.env.example infra/.env

# 2. Generate a strong JWT secret
python -c "import secrets; print(secrets.token_urlsafe(32))"
# Paste the output into infra/.env → JWT_SECRET

# 3. Generate admin password hash
python -c "from passlib.context import CryptContext; print(CryptContext(schemes=['bcrypt']).hash('your-password'))"
# Paste the output into infra/.env → ADMIN_PASSWORD_HASH

# 4. Set admin email (or leave default admin@bankai.local)
# infra/.env → ADMIN_EMAIL

# 5. Set LLM provider keys in infra/.env
# Uncomment/set at least one of: LM_STUDIO_API_KEY, OPENROUTER_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY

# 6. Verify the env file has no blank required fields
grep -E '^(JWT_SECRET|ADMIN_PASSWORD_HASH)=' infra/.env
```

**File reference:** `infra/.env` is gitignored. `infra/.env.example` is the committed template with comments for every field. LLM provider catalog lives at `api/config/llm_providers.yaml` — edit that file to add/remove providers and models; no migration needed.

---

## 3. Service ports reference

| Port | Service | When visible | Notes |
|------|---------|-------------|-------|
| `5432` | Postgres | Always (dev + prod) | `postgresql+asyncpg://postgres:5432/bankai` |
| `6379` | Redis | Always | Celery broker + result backend |
| `6333` | Qdrant | Always | Also serves dashboard UI at `http://localhost:6333/dashboard` |
| `8000` | FastAPI | Always | Swagger at `http://localhost:8000/api/docs` |
| `5173` | Vite dev server | Dev profile only | Hot Module Replacement; proxies `/api` to `http://api:8000` |
| `80` | Nginx | Prod profile only | Serves built SPA + reverse-proxies `/api/*` → `api:8000` |

---

## 4. Run — Dev profile (recommended for development)

Dev profile runs all data services + the API with `--reload`, and you run the frontend separately with HMR.

```bash
# Start all backend services
cd infra && docker compose up -d

# Wait for healthy — all three should be "ok"
curl http://localhost:8000/api/health | python -m json.tool
# Expected: {"status":"ok","db":"ok","redis":"ok","qdrant":"ok"}

# Run the frontend (in a separate terminal)
cd web && npm install && npm run dev
# Open http://localhost:5173
```

### What's running

```bash
docker compose ps
# Expected: postgres (healthy), redis (healthy), qdrant (healthy), api (up), worker (up)
```

### Logs

```bash
docker compose logs -f api       # FastAPI logs (+ reload on code changes)
docker compose logs -f worker    # Celery worker logs
docker compose logs -f postgres  # DB query logs (if dev verbosity enabled)
```

### Stop

```bash
docker compose down             # keep volumes (data persists)
docker compose down -v          # wipe all volumes (PG + Qdrant data lost)
```

### Rebuild after code changes

```bash
docker compose up -d --build    # rebuilds the api.Dockerfile image
```

---

## 5. Run — Prod profile

Prod profile builds the React SPA and serves everything through Nginx on `:80` (single origin, no CORS needed).

```bash
cd infra && docker compose --profile prod up -d --build
# Open http://localhost:80
```

```bash
docker compose --profile prod ps
# Expected: same services + web (running)
```

```bash
docker compose --profile prod down
```

---

## 6. Step-by-step HTTP smoke test

Use these `curl` commands to verify every endpoint manually.

### 6.1 Login and get token

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@bankai.local","password":"your-password"}' \
  | python -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
echo "Token: ${TOKEN:0:20}..."
```

If this returns empty, check your `ADMIN_EMAIL` and password match the hash in `infra/.env`.

### 6.2 Verify token

```bash
curl -s http://localhost:8000/api/auth/me \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
# Expected: {"id":"<uuid>","email":"admin@bankai.local","is_admin":true}
```

### 6.3 List LLM models

```bash
curl -s http://localhost:8000/api/llm-models | python -m json.tool | head -30
# Shows providers with available/unavailable flags
```

### 6.4 Upload a PDF

```bash
DOC_RESP=$(curl -s -X POST http://localhost:8000/api/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@../data/bank-statement-document/Dummy-Bank-Statement.pdf")
echo "$DOC_RESP" | python -m json.tool
# Save the document ID
DOC_ID=$(echo "$DOC_RESP" | python -c "import sys,json; print(json.load(sys.stdin)['id'])")
```

Upload the same PDF again — response will include `"deduplicated":true` and return the existing document row.

### 6.5 Upload rejects non-PDF

```bash
curl -s -X POST http://localhost:8000/api/documents \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/etc/hosts;filename=test.txt" | python -m json.tool
# Expected: 422 {"error":{"code":"VALIDATION_ERROR","message":"Only PDF files accepted"}}
```

### 6.6 Create an agent run (batch)

Pick a working provider + model from step 6.3. Example with LM Studio:

```bash
RUN_RESP=$(curl -s -X POST http://localhost:8000/api/agent-runs \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d "{\"document_ids\":[\"$DOC_ID\"],\"agent\":\"crewai\",\"question\":\"What are the total debits?\",\"llm_provider_id\":\"lm-studio\",\"llm_model_id\":\"openai/qwen2.5-14b-instruct\"}")
echo "$RUN_RESP" | python -m json.tool
RUN_ID=$(echo "$RUN_RESP" | python -c "import sys,json; print(json.load(sys.stdin)['id'])")
```

### 6.7 Poll batch status

```bash
curl -s http://localhost:8000/api/agent-runs/$RUN_ID \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
# Status transitions: pending → running → succeeded | partial | failed
# When succeeded: markdown_report + transactions populated
```

Repeat every ~2s until terminal status. (The frontend does this automatically via `useBatchPolling`.)

### 6.8 List all batches

```bash
curl -s "http://localhost:8000/api/agent-runs?limit=5" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
```

### 6.9 Retry failed items (if any)

```bash
curl -s -X POST "http://localhost:8000/api/agent-runs/$RUN_ID/retry" \
  -H "Authorization: Bearer $TOKEN" | python -m json.tool
# 202 {"retried_item_ids":[...]}
# Or 422 if no failed items
```

### 6.10 Soft-delete batch

```bash
curl -s -X DELETE http://localhost:8000/api/agent-runs/$RUN_ID \
  -H "Authorization: Bearer $TOKEN" -w "%{http_code}"
# Expected: 204
```

### 6.11 Soft-delete document

```bash
curl -s -X DELETE http://localhost:8000/api/documents/$DOC_ID \
  -H "Authorization: Bearer $TOKEN" -w "%{http_code}"
# Expected: 204 (unless referenced by a running item → 409)
```

### 6.12 Health check (no auth needed)

```bash
curl -s http://localhost:8000/api/health | python -m json.tool
```

---

## 7. Testing — Backend (pytest)

### How tests are structured

Tests live at `api/app/tests/`. The conftest mocks `crewai`, `litellm`, `instructor`, and `langchain_*` modules so **most tests run without a real LLM, Qdrant, or Postgres** — only the adapter contract, API validation rules, registry logic, and worker state machine are verified.

### Env vars for tests

Setting the test requires only a subset of production env vars:

```bash
export POSTGRES_USER=test POSTGRES_PASSWORD=test POSTGRES_HOST=localhost
export JWT_SECRET=test-secret ADMIN_EMAIL=admin@test.com ADMIN_PASSWORD_HASH='$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW'
# The hash above is bcrypt('password') — safe for test-only use
```

### Run from host

```bash
cd api && pip install -e ".[dev]"
python -m pytest app/tests/ -v
```

### Run inside the container

```bash
docker compose exec api python -m pytest app/tests/ -v
```

### Run a single test file

```bash
python -m pytest app/tests/test_api_agent_runs.py -v
```

### Test file reference

| File | What it tests | Needs Postgres? |
|------|---------------|-----------------|
| `test_api_auth.py` | Login 401 on bad creds, me endpoint shape | Some tests |
| `test_api_documents.py` | Upload rejects non-PDF, requires auth | No |
| `test_api_agent_runs.py` | Rejects empty doc_ids, unknown agent | No |
| `test_api_llm.py` | `/llm-models` returns providers | No |
| `test_crewai_adapter.py` | Adapter `run()` returns AgentResult with mocked Crew + mocked extractor | No |
| `test_worker_task.py` | Runner status transitions (pending→running→succeeded/failed), run status machine | No (eager Celery + mock adapter) |

---

## 8. Testing — Frontend (vitest)

```bash
cd web && npm install
```

### Run all tests (one-shot)

```bash
npm run test
```

### Watch mode (re-run on changes)

```bash
npm run test:watch
```

### Lint

```bash
npm run lint
```

### Verify production build

```bash
npm run build
```

### Test file reference

| File | What it tests |
|------|---------------|
| `src/__tests__/AgentDropdown.test.tsx` | Renders 3 options; only crewai enabled, others disabled |
| `src/__tests__/ModelDropdown.test.tsx` | Renders grouped dropdown with mocked `useLLMModels`; disabled provider shown with reason |
| `src/__tests__/BatchItemCard.test.tsx` | Renders pending/running/succeeded/failed variants; expand toggles; retry callback |
| `src/__tests__/useBatchPolling.test.ts` | Hook exists and is a function (structural; full behavior test requires React Query provider) |

---

## 9. Troubleshooting

### Container won't start — missing ADMIN_EMAIL or ADMIN_PASSWORD_HASH

The FastAPI lifespan asserts these env vars are set and not empty. Check:

```bash
docker compose logs api | tail -20
# Look for: "ADMIN_EMAIL and ADMIN_PASSWORD_HASH must be set"
```

Fix: confirm `infra/.env` has values for both. If you changed the env file, restart:

```bash
docker compose down && docker compose up -d
```

### LM Studio (local LLM) unreachable from container

The default `llm_providers.yaml` uses `http://host.docker.internal:1234/v1` for LM Studio.

- **macOS / Windows:** Docker Desktop provides `host.docker.internal` automatically.
- **Linux:** Requires Docker 20.10+. If it doesn't resolve, add to `infra/docker-compose.yml`:

```yaml
services:
  api:
    extra_hosts:
      - "host.docker.internal:host-gateway"
  worker:
    extra_hosts:
      - "host.docker.internal:host-gateway"
```

### Port conflicts

Ports already in use on the host:

| Conflict | Fix |
|----------|-----|
| `5432` (postgres) | `export POSTGRES_PORT=5433` in `.env`, or stop the host postgres |
| `6333` (qdrant) | Stop host Qdrant or remap in `docker-compose.override.yml` |
| `8000` (fastapi) | Stop host process or remap |
| `5173` (vite) | Change in `web/vite.config.ts` |
| `80` (nginx) | Requires root or remap in compose |

### Migrations not applied

The API container runs `alembic upgrade head` on startup. If it was skipped or the DB was recreated:

```bash
docker compose exec api alembic -c alembic.ini upgrade head
```

To check current migration state:

```bash
docker compose exec api alembic -c alembic.ini current
```

### Frontend can't reach API in dev

The Vite dev server proxies `/api` to `http://api:8000`. Verify:

```bash
docker compose ps api      # must show "running" (not "starting" or "restarting")
curl http://localhost:8000/api/health  # must return ok
```

If the Vite proxy fails, check `web/vite.config.ts` — `proxy: { "/api": "http://api:8000" }`.

### Qdrant data persists across restarts

The named volume `qdrant_data` survives `docker compose down`. To wipe Qdrant:

```bash
docker compose down -v   # wipes all volumes: pgdata + qdrant_data
```

### Worker tasks stuck in "pending"

Check the worker is running and connected:

```bash
docker compose logs worker | tail -20
# Look for: "celery@<id> ready"
# Errors: "consumer: Cannot connect to redis://redis:6379/0" → redis not healthy yet
```

---

## 10. Cheat sheet

```bash
# --- Docker Compose ---
docker compose ps                          # service status
docker compose logs -f api                 # API logs (follow)
docker compose logs -f worker              # worker logs (follow)
docker compose logs -f api | grep -i error # filter errors
docker compose restart api                 # restart API (picks up code changes if mounted)
docker compose down                        # stop + remove containers (keep volumes)
docker compose down -v                     # stop + remove containers + wipe volumes
docker compose up -d --build               # rebuild image + start

# --- Alembic ---
docker compose exec api alembic -c alembic.ini current          # show current DB revision
docker compose exec api alembic -c alembic.ini upgrade head     # apply all migrations
docker compose exec api alembic -c alembic.ini downgrade -1     # rollback one revision

# --- Backend tests ---
docker compose exec api python -m pytest app/tests/ -v              # all tests
docker compose exec api python -m pytest app/tests/ -v -k "auth"    # filter by name
cd api && python -m pytest app/tests/ -v                            # run on host

# --- Frontend ---
cd web && npm run dev           # Vite HMR on :5173
cd web && npm run test          # vitest one-shot
cd web && npm run test:watch    # vitest watch
cd web && npm run build         # production build check
cd web && npm run lint          # ESLint

# --- Service health ---
curl -s http://localhost:8000/api/health | python -m json.tool
curl -s http://localhost:8000/api/docs   # Swagger UI URL (open in browser)
```

---

## 11. References

- **Full-stack app design spec:** [docs/superpowers/specs/2026-07-25-full-stack-app-design.md](../superpowers/specs/2026-07-25-full-stack-app-design.md)
- **Full-stack app implementation plan:** [docs/superpowers/plans/2026-07-25-full-stack-app.md](../superpowers/plans/2026-07-25-full-stack-app.md)
- **Multi-harness agents guide:** [docs/guides/multi-harness-agents-guide.md](./multi-harness-agents-guide.md) — using the experiment harnesses (Deep Agents, CrewAI notebook, Hermes)
- **Multi-harness agents design:** [docs/superpowers/specs/2026-07-20-multi-harness-agents-design.md](../superpowers/specs/2026-07-20-multi-harness-agents-design.md)
- **Root README roadmap:** [README.md](../../README.md#roadmap)
