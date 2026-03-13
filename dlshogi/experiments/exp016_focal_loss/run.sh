#!/bin/bash
# exp016: Focal Loss for policy head (base: exp015, InceptionNeXt depth=10)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="$(basename "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

source "$REPO_ROOT/.venv/bin/activate"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

cd "$REPO_ROOT"

python -m dlshogi.experiments.exp016_focal_loss.ptl_focal_main fit \
  --config dlshogi/config.yaml \
  --config "$SCRIPT_DIR/config.yaml" \
  --trainer.logger.init_args.name="$EXP_NAME" \
  "$@" \
  2>&1 | tee "$SCRIPT_DIR/fit.log"
