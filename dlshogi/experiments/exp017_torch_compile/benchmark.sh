#!/bin/bash
# Profile torch.compile optimizations

cd "$(dirname "$0")/../../.."
source .venv/bin/activate

python -m dlshogi.experiments.exp017_torch_compile.benchmark "$@"
