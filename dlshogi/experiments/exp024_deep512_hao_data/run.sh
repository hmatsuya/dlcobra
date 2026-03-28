#!/bin/bash
# exp024: Continue exp023 (InceptionNeXt deep512) with hao dataset
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="$(basename "$SCRIPT_DIR")"
DLSHOGI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$DLSHOGI_DIR/.." && pwd)"

source "$REPO_ROOT/.venv/bin/activate"
cd "$DLSHOGI_DIR"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# Extract model weights from exp023 best checkpoint (if not already done)
WEIGHTS="$SCRIPT_DIR/exp023_best.pt"
if [ ! -f "$WEIGHTS" ]; then
  echo "Extracting weights from exp023 best checkpoint..."
  python "$SCRIPT_DIR/extract_weights.py" \
    "$DLSHOGI_DIR/wandb/wcsc36/s3lpnx8n/checkpoints/epoch=0-step=71250.ckpt" \
    "$WEIGHTS"
fi

python "$SCRIPT_DIR/train.py" fit \
  --config config.yaml \
  --config "$SCRIPT_DIR/config.yaml" \
  --trainer.logger.init_args.name="$EXP_NAME" \
  "$@" \
  2>&1 | tee "$SCRIPT_DIR/fit.log"
