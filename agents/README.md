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
PYTHONPATH=. python run_e2e.py --pdf ../../data/bank-statement-document/Dummy-Bank-Statement.pdf --question "Summarize total debits and credits"
```

### Hermes (Docker required)
```bash
cd agents/hermes
./scripts/setup_sandbox.sh
./scripts/run_e2e.sh
```

## Skills policy
Each harness has its **own copy** of domain skills under `skills/`. Edits do not auto-sync.
