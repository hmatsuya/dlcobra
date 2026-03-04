# Tech Stack

## Languages
- Python 3.10+ (training, utilities)
- C++ (core Shogi library, performance-critical code)

## ML Framework
- PyTorch (neural network training)
- PyTorch Lightning (training orchestration)
- ONNX (model export for inference)

## Key Dependencies
```
torch          # Neural networks
lightning      # Training framework (LightningCLI)
wandb          # Experiment tracking
cshogi         # Python Shogi library
h5py           # HDF5 data storage
dask           # Lazy data loading
scipy          # Scientific computing
onnx           # Model export
```

## C++ Dependencies
- CUDA 12.1+ / cuDNN 8.9+ / TensorRT 8.6+ (inference)
- Apery-derived Shogi library (board management, move generation)

## Build Systems
- venv (Python virtual environment)
- Visual Studio 2022 (Windows C++ builds)
- g++ / Make (Linux C++ builds)

## Common Commands

### Python Environment
```bash
source .venv/bin/activate  # Activate virtualenv
pip install -r requirements.txt  # Install dependencies (if needed)
```

### Training
```bash
# Run experiment (venv activation handled by run.sh)
bash dlshogi/experiments/exp001_fewer_activations/run.sh
bash dlshogi/experiments/exp001_fewer_activations/run.sh --debug
```

### Model Export
```bash
python dlshogi/convert_model_to_onnx.py <model_path> <output_path>
```

### C++ Build (Linux)
```bash
cd cppshogi && make
cd usi && make
```

## Configuration
- Training config: `dlshogi/config.yaml` (YAML, LightningCLI format)
- Experiment configs inherit from base and override specific values
- Key hyperparameters: network architecture, batch_size, learning rate schedule
