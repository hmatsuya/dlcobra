#!/bin/bash
# Convert all .bin (PSV) files in the source directory to .hcpe format.
# Usage: bash convert_all_psv.sh [--delete]
#   --delete: delete source .bin files after successful conversion

set -euo pipefail

DELETE_SOURCE=false
if [ "${1:-}" = "--delete" ]; then
    DELETE_SOURCE=true
fi

SRC_DIR="/mnt/nvme0n1p3/Users/hmats/data"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONVERTER="${SCRIPT_DIR}/psv_to_hcpe.py"

if [ ! -d "$SRC_DIR" ]; then
    echo "Error: Source directory not found: $SRC_DIR"
    exit 1
fi

shopt -s nullglob
files=("$SRC_DIR"/*.bin)

if [ ${#files[@]} -eq 0 ]; then
    echo "No .bin files found in $SRC_DIR"
    exit 0
fi

echo "Found ${#files[@]} .bin file(s) to convert"

converted=0
failed=0
skipped=0

for f in "${files[@]}"; do
    base="$(basename "$f" .bin)"
    OUT_DIR="/mnt/nvme1n1p2/data/hao"
    mkdir -p "$OUT_DIR"
    out="${OUT_DIR}/${base}.hcpe"
    # Skip files that aren't valid PSV (must be multiple of 40 bytes)
    size=$(stat -c%s "$f")
    if [ $((size % 40)) -ne 0 ] || [ "$size" -eq 0 ]; then
        echo "Skipping $(basename "$f"): not a valid PSV file (${size} bytes)"
        skipped=$((skipped + 1))
        continue
    fi

    echo "--- Converting: $(basename "$f") ($(( size / 40 )) positions) ---"

    if python "$CONVERTER" "$f" "$out"; then
        rm "$f"
        echo "Converted and deleted: $(basename "$f")"
        converted=$((converted + 1))
    else
        echo "FAILED: $(basename "$f") (kept original)"
        failed=$((failed + 1))
    fi
done

echo ""
echo "Done. Converted: $converted, Skipped: $skipped, Failed: $failed"
