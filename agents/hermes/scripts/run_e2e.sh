#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
PDF_REL="bank-statement-document/Dummy-Bank-Statement.pdf"
PDF_HOST="$REPO/data/$PDF_REL"

"$ROOT/scripts/setup_sandbox.sh"

if [[ ! -f "$PDF_HOST" ]]; then
  echo "Sample PDF missing: $PDF_HOST" >&2
  exit 2
fi

PROMPT=$(cat <<INNER
Run the bank-statement E2E inside the sandbox:
1) Read PDF at /data/${PDF_REL}
2) Parse using bank-statement-parsing skill
3) Redact PII using pii-handling skill
4) Write redacted extract to /workspace/e2e-extract.md
5) Answer: What amounts and balances appear in the statement?
6) Write the final answer only as markdown to /workspace/e2e-answer.md
Follow output-format skill. Do not access paths outside /data and /workspace.
INNER
)

if ! command -v hermes >/dev/null 2>&1; then
  echo "hermes CLI not found on PATH."
  echo "Install from https://hermes-agent.nousresearch.com/ then re-run."
  echo "Documented prompt saved for manual run:"
  printf '%s\n' "$PROMPT" | tee "$ROOT/workspace/e2e-prompt.txt"
  exit 0
fi

# Prefer project config; exact flag names may vary by Hermes version — adjust per `hermes --help`.
export HERMES_WRITE_SAFE_ROOT="${HERMES_WRITE_SAFE_ROOT:-/workspace}"
set +e
hermes chat --config "$ROOT/config.yaml" -q "$PROMPT"
status=$?
set -e
if [[ $status -ne 0 ]]; then
  echo "hermes chat failed (exit $status). Prompt is in workspace/e2e-prompt.txt for manual retry."
  printf '%s\n' "$PROMPT" > "$ROOT/workspace/e2e-prompt.txt"
  exit $status
fi
echo "Hermes E2E finished. Check $ROOT/workspace/"
