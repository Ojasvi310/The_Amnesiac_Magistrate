#!/bin/bash
set -euo pipefail

python -m src.eval.report --all-regimes
python -m src.eval.confusion_set --evaluate
