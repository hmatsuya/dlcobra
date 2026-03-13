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

### Inference Optimization
```bash
# Benchmark with torch.compile (1.2-1.3x speedup, exp017)
python dlshogi/experiments/exp017_torch_compile/benchmark_optimizations.py

# For production: Export to ONNX (additional 2-3x speedup with TensorRT)
python dlshogi/convert_model_to_onnx.py <model_path> <output_path>
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

## Inference Optimization

### torch.compile (推奨)
PyTorch 2.0+のJITコンパイラによる推論高速化:
- 速度: 1.22-1.30x高速化（exp017で検証済み）
- 精度: 変化なし
- 使用方法:
```python
model = PolicyValueNetwork().to(device).eval()
model = torch.compile(model, mode="reduce-overhead")
```

### ONNX + TensorRT (本番デプロイ)
更なる高速化が必要な場合:
- 速度: torch.compile比で更に2-3x高速化
- 合計: ベースラインから約3-4x高速化
- FP16精度で実行可能

## Configuration
- Training config: `dlshogi/config.yaml` (YAML, LightningCLI format)
- Experiment configs inherit from base and override specific values
- Key hyperparameters: network architecture, batch_size, learning rate schedule
