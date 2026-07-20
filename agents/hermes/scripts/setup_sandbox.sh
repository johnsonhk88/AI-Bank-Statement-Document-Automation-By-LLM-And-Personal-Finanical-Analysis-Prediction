#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"

"$ROOT/scripts/check_config.sh"

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is required for Hermes sandbox" >&2
  exit 1
fi
docker info >/dev/null

mkdir -p "$ROOT/workspace"
if [[ ! -d "$REPO/data" ]]; then
  echo "Missing repo data/: $REPO/data" >&2
  exit 1
fi

# Document env expected at runtime
cat > "$ROOT/workspace/.sandbox-env.example" <<INNER
# Export on host before run_e2e.sh (do not commit real secrets)
# export HERMES_WRITE_SAFE_ROOT=/workspace
# Ensure Hermes uses project config with terminal.backend=docker
INNER

echo "Sandbox prerequisites OK"
echo "  data (RO):  $REPO/data -> /data"
echo "  workspace:  $ROOT/workspace -> /workspace"
echo "  skills:     $ROOT/skills"
