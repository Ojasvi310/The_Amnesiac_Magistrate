#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$REPO_ROOT"

echo "Building Docker image..."
docker build -t continual-counsel-test .

# Run the import graph check
echo "Checking import constraints in src/online/..."
if grep -r "import requests\|import httpx\|import torch\|import peft\|import bitsandbytes\|from transformers import" src/online/; then
    echo "FAIL: Prohibited imports found in src/online/ (no training or network dependencies allowed)."
    exit 1
fi
echo "Import constraints verified."

echo "Starting container with network disabled..."
# Run container in background with --network none
CONTAINER_ID=$(docker run -d --rm --network none -p 8000:8000 continual-counsel-test)

# Cleanup on exit
trap "docker stop $CONTAINER_ID" EXIT

echo "Waiting for service to start..."
sleep 5

echo "Testing health endpoint from host loopback..."
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "Failed")

if [[ "$HTTP_STATUS" == "200" ]]; then
    echo "Smoke test PASSED — no outbound network required"
    exit 0
else
    echo "Smoke test FAILED — received HTTP $HTTP_STATUS"
    docker logs "$CONTAINER_ID"
    exit 1
fi
