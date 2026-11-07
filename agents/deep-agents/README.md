# Deep Agents harness

LangChain **deepagents** with agentskills.io skills for bank-statement automation.

**Full guide:** [docs/guides/multi-harness-agents-guide.md](../../docs/guides/multi-harness-agents-guide.md)

## Setup

```bash
cd agents/deep-agents
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Test

```bash
PYTHONPATH=. pytest tests/ -v
# expect: 6 passed
```

## E2E (no LLM — deterministic tools)

```bash
PYTHONPATH=. python run_e2e.py \
  --pdf ../../data/samples/Dummy-Bank-Statement.pdf \
  --question "What amounts appear in the statement?" \
  --mode pipeline
```

## E2E (LLM agent)

Requires a local/remote model reachable by the deepagents model string (needs `langchain-ollama` for `ollama:` models):

```bash
PYTHONPATH=. python run_e2e.py \
  --pdf ../../data/samples/Dummy-Bank-Statement.pdf \
  --question "Summarize total debits and credits" \
  --mode agent \
  --model ollama:llama3.2
```

Skills live in `./skills/` (synced from repo-root `skills/` via `python scripts/sync_skills.py --apply`). Outputs go to `./workspace/` (gitignored).
