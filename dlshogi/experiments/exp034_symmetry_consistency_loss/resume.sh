#!/bin/bash
# Resume exp034 from last checkpoint
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="$(basename "$SCRIPT_DIR")"
DLSHOGI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$DLSHOGI_DIR/.." && pwd)"

source "$REPO_ROOT/.venv/bin/activate"
cd "$DLSHOGI_DIR"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

RUN_ID=$(cat "$SCRIPT_DIR/wandb_run_id" 2>/dev/null)
if [ -z "$RUN_ID" ]; then
  echo "Error: wandb_run_id not found. Run run.sh first."
  exit 1
fi

CKPT="$DLSHOGI_DIR/wandb/wcsc36/$RUN_ID/checkpoints/last.ckpt"
if [ ! -f "$CKPT" ]; then
  echo "Error: checkpoint not found: $CKPT"
  exit 1
fi

python ptl.py fit \
  --config config.yaml \
  --config "$SCRIPT_DIR/config.yaml" \
  --trainer.logger.init_args.name="$EXP_NAME" \
  --ckpt_path="$CKPT" \
  "$@" \
  2>&1 | tee -a "$SCRIPT_DIR/fit.log"
