#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$ROOT/config.yaml"
if [[ ! -f "$CFG" ]]; then
  echo "Missing config: $CFG" >&2
  exit 1
fi
if grep -E '^\s*backend:\s*local\s*$' "$CFG" >/dev/null; then
  echo "REFUSE: terminal.backend must not be local" >&2
  exit 1
fi
if ! grep -E '^\s*backend:\s*docker\s*$' "$CFG" >/dev/null; then
  echo "REFUSE: terminal.backend must be docker" >&2
  exit 1
fi
echo "config ok: docker backend"
