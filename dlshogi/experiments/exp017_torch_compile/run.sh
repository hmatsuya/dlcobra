#!/bin/bash
# Training script for exp017
# Note: This experiment is for profiling only - training not expected to run

cd "$(dirname "$0")/../../.."
source .venv/bin/activate

if [ "$1" = "--debug" ]; then
    python -m dlshogi.ptl fit \
        --config dlshogi/config.yaml \
        --config dlshogi/experiments/exp017_torch_compile/config.yaml \
        --trainer.fast_dev_run=5
else
    python -m dlshogi.ptl fit \
        --config dlshogi/config.yaml \
        --config dlshogi/experiments/exp017_torch_compile/config.yaml
fi
