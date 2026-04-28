#!/bin/bash
# Step 1: Run structured pruning on exp029 best checkpoint.
# Produces pruned_state_dict.pt in this directory.
#
# Usage:
#   bash dlshogi/experiments/exp032_structured_pruning/prune.sh [--prune-ratio 0.25] [--dry-run]
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DLSHOGI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$DLSHOGI_DIR/.." && pwd)"

source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

python "$SCRIPT_DIR/prune.py" \
  --ckpt "/home/hmatsuya/workspace/Shogi/dlcobra/dlshogi/wandb/wcsc36/wkpn9zzz/checkpoints/epoch=0-step=116250.ckpt" \
  --output "$SCRIPT_DIR/pruned_state_dict.pt" \
  "$@"
