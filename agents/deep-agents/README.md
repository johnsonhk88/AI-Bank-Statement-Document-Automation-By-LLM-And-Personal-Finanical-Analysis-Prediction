# Deep Agents harness

LangChain **deepagents** with agentskills.io skills for bank-statement automation.

## Setup

```bash
cd agents/deep-agents
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## E2E (no LLM — deterministic tools)

```bash
PYTHONPATH=. python run_e2e.py \
  --pdf ../../data/bank-statement-document/Dummy-Bank-Statement.pdf \
  --question "What amounts appear in the statement?" \
  --mode pipeline
```

## E2E (LLM agent)

Requires a local/remote model reachable by the deepagents model string:

```bash
PYTHONPATH=. python run_e2e.py \
  --pdf ../../data/bank-statement-document/Dummy-Bank-Statement.pdf \
  --question "Summarize total debits and credits" \
  --mode agent \
  --model ollama:llama3.2
```

Skills live in `./skills/`. Outputs go to `./workspace/` (gitignored).
