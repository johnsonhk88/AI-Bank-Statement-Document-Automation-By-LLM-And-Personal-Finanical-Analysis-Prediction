# Hermes harness (Docker sandbox)

Nous Research **Hermes Agent** profile for this repo. Tool/shell execution must use **Docker**, not the host shell.

## Security model

| Mount | Container path | Mode |
|-------|----------------|------|
| repo `data/` | `/data` | **read-only** |
| `agents/hermes/workspace/` | `/workspace` | read-write |
| `agents/hermes/skills/` | `/skills` | **read-only** |

**Isolation source of truth:** `docker-compose.yml` documents the intended isolation model.
Hermes manages its own Docker backend — operators **must** configure Hermes docker mounts to match that file:

| Host path | Container | Flags |
|-----------|-----------|-------|
| repo `data/` | `/data` | `:ro` |
| `agents/hermes/workspace/` | `/workspace` | `:rw` |
| `agents/hermes/skills/` | `/skills` | `:ro` |

- `config.yaml` sets `terminal.backend: docker` (enforced by `scripts/check_config.sh`).
- `HERMES_WRITE_SAFE_ROOT=/workspace`
- No secrets in git; use host `~/.hermes/.env`.

### Mount verification checklist

1. Confirm `docker-compose.yml` volumes are: `data→/data:ro`, `workspace→/workspace:rw`, `skills→/skills:ro`.
2. Configure Hermes CLI/docker backend mounts to the **same** three paths and modes (do not rely on compose alone if Hermes spawns its own container).
3. Run `./scripts/setup_sandbox.sh` — it prints required mount flags and checks host paths exist.
4. Inside a sandbox shell: `ls /data` (statements visible), `touch /workspace/write-test` (OK), `touch /data/x` (must fail — read-only).
5. Confirm skills are readable at `/skills` and not writable.

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
