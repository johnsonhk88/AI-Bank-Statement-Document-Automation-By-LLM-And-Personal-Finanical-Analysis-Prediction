# Hermes harness (Docker sandbox)

Nous Research **Hermes Agent** profile for this repo. Tool/shell execution must use **Docker**, not the host shell.

## Security model

| Mount | Container path | Mode |
|-------|----------------|------|
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
