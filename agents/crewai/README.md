# CrewAI harness (baseline)

**Full guide:** [docs/guides/multi-harness-agents-guide.md](../../docs/guides/multi-harness-agents-guide.md) (section CrewAI)

- Notebook: `notebooks/ai_agent_skills_dev.ipynb`
- Canonical skills: `skills/` (repo root) + `skills/crewai_skills_loader.py`
- This folder's `skills/` is a **generated copy**. Edit the root `skills/` tree, then `python scripts/sync_skills.py --apply`.

The production API (`api/app/agents/crewai/adapter.py`) loads canonical `skills/` directly.

## Run baseline

From repo root, with the project venv that has `crewai==1.14.7`:

```bash
jupyter notebook notebooks/ai_agent_skills_dev.ipynb
```
