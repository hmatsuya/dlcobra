#!/bin/bash
# Log disk usage of key project directories
# Usage: bash dlshogi/utils/disk_usage.sh [>> disk_usage.log]
#
# Run before and after training to measure disk consumption.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "=== Disk Usage Snapshot: $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "Project root: $PROJECT_ROOT"
echo ""

# Overall project size
echo "--- Project Total ---"
du -sh "$PROJECT_ROOT" 2>/dev/null

echo ""
echo "--- Key Directories ---"

# Directories that grow during training
dirs=(
    "lightning_logs"
    "dlshogi/lightning_logs"
    "wandb"
    "dlshogi/download"
    "cppshogi/obj"
    "build"
    "build_onnx"
)

for d in "${dirs[@]}"; do
    path="$PROJECT_ROOT/$d"
    if [ -d "$path" ]; then
        du -sh "$path"
    fi
done

echo ""
echo "--- Checkpoints (*.ckpt) ---"
find "$PROJECT_ROOT" -name "*.ckpt" -exec du -sh {} \; 2>/dev/null | sort -rh | head -20

echo ""
echo "--- Disk Free ---"
df -h "$PROJECT_ROOT" | tail -1

echo ""
echo "==========================================="
