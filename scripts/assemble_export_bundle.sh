#!/bin/bash
# Assembles the export bundle for a completed regime update on the Colab side.
# Run this after run_quarter_update_colab.sh completes.
#
# Finds the most recent export bundle directory under exports/, writes manifest.sha256,
# zips the bundle into a single archive, and optionally copies it to Google Drive.
#
# The actual manifest.json and file contents are written by src/offline/export.py;
# this script just archives the result and makes the commit hash explicit.
set -euo pipefail

REGIME="${1:-}"
if [[ -z "$REGIME" ]]; then
    echo "Usage: $0 <regime_name> [<drive_dest_dir>]"
    exit 1
fi
DRIVE_DEST="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
EXPORTS_DIR="$REPO_ROOT/exports"

# Find the most recent bundle dir for this regime (directories only, not .tar.gz files)
BUNDLE_DIR=$(ls -dt "$EXPORTS_DIR/${REGIME}_"* 2>/dev/null | grep -v '\.tar\.gz' | head -1)
if [[ -z "$BUNDLE_DIR" ]]; then
    echo "ERROR: No export bundle found under $EXPORTS_DIR for regime '$REGIME'."
    echo "       Run run_quarter_update_colab.sh first, or check that export.cadence is not 'merge_only'."
    exit 1
fi

BUNDLE_NAME="$(basename "$BUNDLE_DIR")"
# Strip any accidental .tar.gz suffix from the bundle name (defensive)
BUNDLE_NAME="${BUNDLE_NAME%.tar.gz}"
ARCHIVE="$EXPORTS_DIR/${BUNDLE_NAME}.tar.gz"

# Stamp the commit hash into the metadata if export.py hasn't already
COMMIT=$(git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown")
if [[ -f "$BUNDLE_DIR/metadata.json" ]]; then
    # Verify commit_hash field is present; if missing, patch it in via python
    python3 -c "
import json, sys
path = sys.argv[1]; commit = sys.argv[2]
m = json.loads(open(path).read())
if not m.get('commit_hash'):
    m['commit_hash'] = commit
    open(path, 'w').write(json.dumps(m, indent=2))
    print('Patched commit_hash into metadata.json')
" "$BUNDLE_DIR/metadata.json" "$COMMIT"
else
    echo "WARNING: metadata.json not found in bundle. export.py may not have run Step F."
fi

echo "Archiving $BUNDLE_NAME ..."
tar -czf "$ARCHIVE" -C "$EXPORTS_DIR" "$BUNDLE_NAME"
ARCHIVE_SIZE=$(du -sh "$ARCHIVE" | cut -f1)
echo "Archive: $ARCHIVE ($ARCHIVE_SIZE)"

if [[ -n "$DRIVE_DEST" ]]; then
    if [[ -d "$DRIVE_DEST" ]]; then
        cp "$ARCHIVE" "$DRIVE_DEST/"
        echo "Copied to Drive: $DRIVE_DEST/$BUNDLE_NAME.tar.gz"
    else
        echo "WARNING: Drive destination '$DRIVE_DEST' not found. Archive left at $ARCHIVE."
        echo "         Copy it manually or download via Colab's Files panel."
    fi
else
    echo "No Drive destination specified. Archive is at:"
    echo "  $ARCHIVE"
    echo "Download it via Colab Files panel or supply a Drive path as \$2."
fi

echo "Done. Bundle commit hash: $COMMIT"
