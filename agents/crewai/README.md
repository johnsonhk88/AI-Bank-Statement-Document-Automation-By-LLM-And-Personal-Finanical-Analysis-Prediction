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
