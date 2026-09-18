#!/usr/bin/env bash
# verify_export_bundle.sh
# Verify the integrity of an export bundle before registering it.
#
# Usage: ./scripts/verify_export_bundle.sh <bundle_dir>
#
# Requirements: sha256sum, jq, python3 (all standard on modern Linux/macOS).
# On macOS, sha256sum may not be installed; use: brew install coreutils

set -euo pipefail

BUNDLE_DIR="${1:-}"
if [[ -z "$BUNDLE_DIR" ]]; then
    echo "Usage: $0 <bundle_dir>" >&2
    exit 1
fi

if [[ ! -d "$BUNDLE_DIR" ]]; then
    echo "ERROR: $BUNDLE_DIR is not a directory." >&2
    exit 1
fi

MANIFEST="$BUNDLE_DIR/manifest.json"
MANIFEST_SHA256="$BUNDLE_DIR/manifest.sha256"

if [[ ! -f "$MANIFEST" ]]; then
    echo "ERROR: manifest.json not found in $BUNDLE_DIR" >&2
    exit 1
fi

if [[ ! -f "$MANIFEST_SHA256" ]]; then
    echo "ERROR: manifest.sha256 not found in $BUNDLE_DIR" >&2
    exit 1
fi

# ---- 1. Verify manifest.json against manifest.sha256 -------------------------
echo "Verifying manifest integrity..."

COMPUTED_MANIFEST_HASH=$(sha256sum "$MANIFEST" | awk '{print $1}')
CLAIMED_MANIFEST_HASH=$(cat "$MANIFEST_SHA256" | tr -d '[:space:]')

if [[ "$COMPUTED_MANIFEST_HASH" != "$CLAIMED_MANIFEST_HASH" ]]; then
    echo "FAIL: manifest.json hash mismatch"
    echo "  computed : $COMPUTED_MANIFEST_HASH"
    echo "  claimed  : $CLAIMED_MANIFEST_HASH"
    echo "BUNDLE VERIFICATION FAILED -- rejecting"
    exit 1
fi

echo "PASS: manifest.json integrity verified"

# ---- 2. Verify each file listed in manifest.json ----------------------------
OVERALL_PASS=true
FILE_COUNT=$(jq '.files | length' "$MANIFEST")

echo "Verifying $FILE_COUNT file(s) listed in manifest..."

for i in $(seq 0 $((FILE_COUNT - 1))); do
    REL_PATH=$(jq -r ".files[$i].path" "$MANIFEST")
    EXPECTED_HASH=$(jq -r ".files[$i].sha256" "$MANIFEST")
    EXPECTED_SIZE=$(jq -r ".files[$i].size_bytes" "$MANIFEST")

    FULL_PATH="$BUNDLE_DIR/$REL_PATH"

    if [[ ! -f "$FULL_PATH" ]]; then
        echo "FAIL [$REL_PATH]: file does not exist"
        OVERALL_PASS=false
        continue
    fi

    # Check byte size
    ACTUAL_SIZE=$(wc -c < "$FULL_PATH" | tr -d ' ')
    if [[ "$ACTUAL_SIZE" != "$EXPECTED_SIZE" ]]; then
        echo "FAIL [$REL_PATH]: size mismatch (expected $EXPECTED_SIZE, got $ACTUAL_SIZE)"
        OVERALL_PASS=false
        # Continue to also check hash so the operator sees the full picture.
    fi

    # Check SHA-256
    ACTUAL_HASH=$(sha256sum "$FULL_PATH" | awk '{print $1}')
    if [[ "$ACTUAL_HASH" != "$EXPECTED_HASH" ]]; then
        echo "FAIL [$REL_PATH]: SHA-256 mismatch"
        echo "  expected : $EXPECTED_HASH"
        echo "  computed : $ACTUAL_HASH"
        OVERALL_PASS=false
    else
        echo "PASS [$REL_PATH]"
    fi
done

# ---- 3. Overall verdict ------------------------------------------------------
if [[ "$OVERALL_PASS" != "true" ]]; then
    echo ""
    echo "BUNDLE VERIFICATION FAILED -- rejecting"
    exit 1
fi

echo ""
echo "Bundle verified. Registering adapter..."
python3 -m src.audit.registry register --bundle-dir "$BUNDLE_DIR"
