# Canonical skills

Edit **only** the `SKILL.md` files in this directory.

Then copy them into each agent harness:

```bash
python scripts/sync_skills.py --apply
python scripts/sync_skills.py --check
```

The production API (`api/app/agents/crewai/adapter.py`) loads this directory directly via `crewai_skills_loader.py`. Harness copies under `agents/*/skills/` are generated — do not edit them by hand.
