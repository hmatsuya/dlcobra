#!/bin/bash
# exp032: Structured pruning fine-tuning
# Step 2: Fine-tune the pruned model with KD from exp029 best ckpt.
#
# Prerequisites: run prune.sh first to generate pruned_state_dict.pt
#
# Usage:
#   bash dlshogi/experiments/exp032_structured_pruning/run.sh [--debug]
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="$(basename "$SCRIPT_DIR")"
DLSHOGI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$DLSHOGI_DIR/.." && pwd)"

# Verify pruned weights exist before starting training
PRUNED_WEIGHTS="$SCRIPT_DIR/pruned_state_dict.pt"
if [ ! -f "$PRUNED_WEIGHTS" ]; then
  echo "Error: pruned_state_dict.pt not found."
  echo "Run first: bash $SCRIPT_DIR/prune.sh"
  exit 1
fi

source "$REPO_ROOT/.venv/bin/activate"
cd "$DLSHOGI_DIR"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

python ptl.py fit \
  --config config.yaml \
  --config "$SCRIPT_DIR/config.yaml" \
  --trainer.logger.init_args.name="$EXP_NAME" \
  "$@" \
  2>&1 | tee "$SCRIPT_DIR/fit.log"

# Save the WandB run ID so resume.sh can find the correct checkpoint.
WANDB_DIR="$DLSHOGI_DIR/wandb/wcsc36"
if [ -d "$WANDB_DIR" ]; then
  LATEST_ID=""
  LATEST_MTIME=0
  for run_dir in "$WANDB_DIR"/*/; do
    ckpt="$run_dir/checkpoints/last.ckpt"
    if [ -f "$ckpt" ]; then
      mtime=$(stat -c %Y "$ckpt" 2>/dev/null || stat -f %m "$ckpt" 2>/dev/null)
      if [ "$mtime" -gt "$LATEST_MTIME" ]; then
        LATEST_MTIME=$mtime
        LATEST_ID="$(basename "$run_dir")"
      fi
    fi
  done
  if [ -n "$LATEST_ID" ]; then
    echo "$LATEST_ID" > "$SCRIPT_DIR/wandb_run_id"
    echo "Saved WandB run ID: $LATEST_ID"
  fi
fi
