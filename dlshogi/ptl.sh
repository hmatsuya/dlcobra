#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../.venv/bin/activate"
cd "$SCRIPT_DIR"
python ptl.py fit --config config.yaml 2>&1 | tee fit.log