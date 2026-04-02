#!/bin/bash
# exp027: exp022 + horizontal flip data augmentation (50%)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="$(basename "$SCRIPT_DIR")"
DLSHOGI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$DLSHOGI_DIR/.." && pwd)"

source "$REPO_ROOT/.venv/bin/activate"
cd "$DLSHOGI_DIR"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

python ptl.py fit \
  --config config.yaml \
  --config "$SCRIPT_DIR/config.yaml" \
  --trainer.logger.init_args.name="$EXP_NAME" \
  "$@" \
  2>&1 | tee "$SCRIPT_DIR/fit.log"
