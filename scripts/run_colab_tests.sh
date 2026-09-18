#!/bin/bash
set -euo pipefail

# Runs Colab-only tests (which require GPU and training deps)
pip install -r requirements-offline.txt
pytest -m colab_only -v tests/
