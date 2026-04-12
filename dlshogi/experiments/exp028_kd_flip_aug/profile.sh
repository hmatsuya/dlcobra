#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DLSHOGI_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$DLSHOGI_DIR/.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"
cd "$DLSHOGI_DIR"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
python "$SCRIPT_DIR/profile.py" "$@"
