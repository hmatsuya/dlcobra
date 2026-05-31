#!/bin/bash
# Resume training from the last checkpoint.
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
  echo "Error: wandb_run_id not found. Run run.sh first."
  exit 1
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
  --ckpt_path "$CKPT_PATH" \
  "$@" \
  2>&1 | tee -a "$SCRIPT_DIR/fit.log"
