# Agent harnesses

Parallel agent frameworks for bank-statement automation. Shared inputs: repo `data/`.

**Full step-by-step (setup · use · test):**  
[docs/guides/multi-harness-agents-guide.md](../docs/guides/multi-harness-agents-guide.md)

| Harness | Path | Skills | Runtime |
|---------|------|--------|---------|
| CrewAI (baseline) | `crewai/` + `backend/app/core/` | CrewAI Skills | project venv + notebook |
| Deep Agents | `deep-agents/` | agentskills.io | local `.venv` + `run_e2e.py` |
| Hermes | `hermes/` | agentskills.io | Hermes CLI + **Docker sandbox** |

## Quick start

### CrewAI baseline

See [crewai/README.md](crewai/README.md) and run:

```bash
jupyter notebook backend/app/core/ai_agent_skills_dev.ipynb
```

### Deep Agents (recommended first — works without LLM)

```bash
cd agents/deep-agents
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. pytest tests/ -v
PYTHONPATH=. python run_e2e.py \
  --pdf ../../data/bank-statement-document/Dummy-Bank-Statement.pdf \
  --question "Summarize total debits and credits" \
  --mode pipeline
```

Details: [deep-agents/README.md](deep-agents/README.md)

### Hermes (Docker required)

```bash
cd agents/hermes
./scripts/setup_sandbox.sh
./scripts/run_e2e.sh
```

Details: [hermes/README.md](hermes/README.md)

## Skills policy

Each harness has its **own copy** of domain skills under `skills/`. Edits do not auto-sync.

## Design

[docs/superpowers/specs/2026-07-20-multi-harness-agents-design.md](../docs/superpowers/specs/2026-07-20-multi-harness-agents-design.md)
