#!/bin/bash
set -e

VENV_DIR=".venv"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

pip install --upgrade pip setuptools wheel
pip install torch lightning scipy cshogi h5py dask pandas icecream onnx pyyaml wandb "jsonargparse[signatures]>=4.27.7"

python setup.py clean --all
rm -rf build dist dlshogi.egg-info
pip install . --force-reinstall
python -c "import dlshogi; print(dlshogi.__version__)"
