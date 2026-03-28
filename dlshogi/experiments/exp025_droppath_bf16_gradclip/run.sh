#!/bin/bash
# exp025: exp023 + DropPath(0.2) + bf16-mixed + gradient_clip_val=1.0
# Start from exp023 best checkpoint (val/loss=2.111, step=71250)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="$(basename "$SCRIPT_DIR")"
DLSHOGI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$DLSHOGI_DIR/.." && pwd)"

source "$REPO_ROOT/.venv/bin/activate"
cd "$DLSHOGI_DIR"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# exp023 best checkpoint (val/loss=2.111, step=71250)
CKPT_PATH="$DLSHOGI_DIR/wandb/wcsc36/s3lpnx8n/checkpoints/epoch=0-step=71250.ckpt"

if [ ! -f "$CKPT_PATH" ]; then
  echo "Error: exp023 best checkpoint not found: $CKPT_PATH"
  exit 1
fi

python ptl.py fit \
  --config config.yaml \
  --config "$SCRIPT_DIR/config.yaml" \
  --trainer.logger.init_args.name="$EXP_NAME" \
  --ckpt_path "$CKPT_PATH" \
  "$@" \
  2>&1 | tee "$SCRIPT_DIR/fit.log"
