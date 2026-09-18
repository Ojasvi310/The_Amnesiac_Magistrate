#!/bin/bash
# Thin wrapper — invoked from within a Colab session after cloning the repo
# and installing requirements-offline.txt. Calls pipeline.py with the right
# arguments. All logic lives in src/offline/pipeline.py, not here.
set -euo pipefail

REGIME="${1:-}"
if [[ -z "$REGIME" ]]; then
    echo "Usage: $0 <regime_name> [--profile proxy|full_scale] [--input-bundle <dir>]"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== Continual Counsel: Quarter Update ==="
echo "Regime: $REGIME"
echo "Repo:   $REPO_ROOT"
echo "Commit: $(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo 'unknown')"
echo "Python: $(python3 --version)"

cd "$REPO_ROOT"
python3 -m src.offline.pipeline \
    --regime "$REGIME" \
    "${@:2}"
