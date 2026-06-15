#!/bin/bash
# exp036: Resume exp029 with longer patience (50 vs 20)
# Uses --ckpt_path to restore full training state (LR, optimizer, step counter).
#
# NOTE: exp029's last.ckpt was saved in a *stopped* state
# (EarlyStopping wait_count=20, patience=20, reason=PATIENCE_EXHAUSTED). Because
# Lightning restores patience/wait_count from the checkpoint, resuming directly
# would re-trigger early stopping after a single non-improving validation,
# ignoring the patience=50 in config.yaml. We therefore resume from a *patched*
# copy whose EarlyStopping state is reset (wait_count=0, patience=50,
# reason=NOT_STOPPED) while model weights, optimizer, LR scheduler, and step
# counter are left untouched. See patch_ckpt_patience.py.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXP_NAME="$(basename "$SCRIPT_DIR")"
DLSHOGI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$DLSHOGI_DIR/.." && pwd)"

# exp029's last checkpoint (step=141250, full Lightning state)
SRC_CKPT_PATH="$DLSHOGI_DIR/wandb/wcsc36/wkpn9zzz/checkpoints/last.ckpt"
# Patched copy with EarlyStopping reset to patience=50 (created on demand)
CKPT_PATH="$DLSHOGI_DIR/wandb/wcsc36/wkpn9zzz/checkpoints/last_patience50.ckpt"

if [ ! -f "$SRC_CKPT_PATH" ]; then
  echo "Error: exp029 checkpoint not found: $SRC_CKPT_PATH"
  exit 1
fi

source "$REPO_ROOT/.venv/bin/activate"
cd "$DLSHOGI_DIR"

export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

# Create the patched checkpoint if it doesn't already exist
if [ ! -f "$CKPT_PATH" ]; then
  echo "Patched checkpoint not found; creating it from exp029 last.ckpt"
  python "$SCRIPT_DIR/patch_ckpt_patience.py" "$SRC_CKPT_PATH" "$CKPT_PATH" --patience 50
fi

echo "Resuming from patched exp029 ckpt (step=141250, LR=0.000555, patience=50)"

python ptl.py fit \
  --config config.yaml \
  --config "$SCRIPT_DIR/config.yaml" \
  --trainer.logger.init_args.name="$EXP_NAME" \
  --ckpt_path "$CKPT_PATH" \
  "$@" \
  2>&1 | tee "$SCRIPT_DIR/fit.log"

# Save the WandB run ID
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
