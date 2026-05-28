#!/bin/bash
# Resume training from the last checkpoint.
# Usage: bash dlshogi/experiments/exp035_.../resume.sh [extra args...]
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="$(basename "$SCRIPT_DIR")"
DLSHOGI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$DLSHOGI_DIR/.." && pwd)"
WANDB_DIR="$DLSHOGI_DIR/wandb/wcsc36"

RUN_ID_FILE="$SCRIPT_DIR/wandb_run_id"
if [ -f "$RUN_ID_FILE" ]; then
  RUN_ID="$(cat "$RUN_ID_FILE")"
  CKPT_PATH="$WANDB_DIR/$RUN_ID/checkpoints/last.ckpt"
  if [ ! -f "$CKPT_PATH" ]; then
    echo "Error: checkpoint not found for run $RUN_ID: $CKPT_PATH"
    exit 1
  fi
else
  echo "Warning: wandb_run_id not found, falling back to mtime scan (may pick wrong experiment)"
  if [ ! -d "$WANDB_DIR" ]; then
    echo "Error: wandb directory not found: $WANDB_DIR"
    exit 1
  fi
  CKPT_PATH=""
  LATEST_MTIME=0
  for run_dir in "$WANDB_DIR"/*/; do
    ckpt="$run_dir/checkpoints/last.ckpt"
    if [ -f "$ckpt" ]; then
      mtime=$(stat -c %Y "$ckpt" 2>/dev/null || stat -f %m "$ckpt" 2>/dev/null)
      if [ "$mtime" -gt "$LATEST_MTIME" ]; then
        LATEST_MTIME=$mtime
        CKPT_PATH="$ckpt"
        RUN_ID="$(basename "$run_dir")"
      fi
    fi
  done
  if [ -z "$CKPT_PATH" ]; then
    echo "Error: No last.ckpt found in $WANDB_DIR"
    exit 1
  fi
fi

echo "Resuming $EXP_NAME from: $CKPT_PATH (run: $RUN_ID)"

source "$REPO_ROOT/.venv/bin/activate"
cd "$DLSHOGI_DIR"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

python ptl.py fit \
  --config config.yaml \
  --config "$SCRIPT_DIR/config.yaml" \
  --trainer.logger.init_args.name="$EXP_NAME" \
  --trainer.logger.init_args.id="$RUN_ID" \
  --trainer.logger.init_args.resume="must" \
  --trainer.callbacks[2].init_args.patience=80 \
  --ckpt_path "$CKPT_PATH" \
  "$@" \
  2>&1 | tee -a "$SCRIPT_DIR/fit.log"
