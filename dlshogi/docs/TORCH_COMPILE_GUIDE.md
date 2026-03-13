# torch.compile Usage Guide

## Overview

PyTorch 2.0+ includes `torch.compile`, a JIT compiler that provides 1.2-1.3x inference speedup with zero accuracy loss (verified in exp017).

## Quick Start

### For Profiling

```bash
# Profile with torch.compile
python profile.py --compile

# Compare baseline vs compiled
python profile.py                    # Baseline
python profile.py --compile          # With torch.compile
```

### For Inference Code

```python
import torch
from dlshogi.experiments.exp015_inceptionnext_depth10.model import PolicyValueNetwork

# Load model
model = PolicyValueNetwork().to(device).eval()

# Apply torch.compile
model = torch.compile(model, mode="reduce-overhead")

# Use normally
with torch.no_grad():
    policy, value = model(x1, x2)
```

## Benchmark Results (exp017)

Tested on NVIDIA GeForce RTX 3090 with PyTorch 2.10.0:

| Batch Size | Baseline | torch.compile | Speedup |
|------------|----------|---------------|---------|
| 128        | 7.6 ms   | 5.9 ms        | 1.30x   |
| 256        | 14.6 ms  | 11.8 ms       | 1.24x   |
| 512        | 28.3 ms  | 22.9 ms       | 1.23x   |
| 1024       | 55.5 ms  | 45.5 ms       | 1.22x   |
| 2048       | 110.4 ms | 90.6 ms       | 1.22x   |

## Compilation Modes

PyTorch offers several compilation modes:

```python
# Recommended: Best balance of speed and compilation time
model = torch.compile(model, mode="reduce-overhead")

# Alternative modes:
model = torch.compile(model, mode="default")        # Balanced
model = torch.compile(model, mode="max-autotune")   # Slowest compile, fastest runtime
```

For dlshogi, `mode="reduce-overhead"` is recommended.

## When to Use

### ✓ Use torch.compile for:
- Inference/evaluation
- Profiling experiments
- Production USI engine
- MCTS search (batch=128)

### ✗ Don't use torch.compile for:
- Training (compilation overhead on first epoch)
- Debugging (harder to trace errors)
- Rapid prototyping (compilation takes time)

## Integration with Existing Code

### Profiling Scripts

The template `profile.py` now supports `--compile` flag:

```bash
# Quick profiling
bash profile.sh                      # Baseline
python profile.py --compile          # With torch.compile

# Custom batch size
python profile.py --batch-size 256 --compile
```

### Training Scripts

Not recommended for training due to compilation overhead. Use only for final evaluation:

```python
# After training
model.eval()
model = torch.compile(model, mode="reduce-overhead")

# Evaluate on test set
test_accuracy = evaluate(model, test_loader)
```

## Combining with Other Optimizations

### torch.compile + ONNX + TensorRT

For maximum inference speed:

1. **Development**: Use torch.compile (1.2-1.3x speedup)
   ```python
   model = torch.compile(model, mode="reduce-overhead")
   ```

2. **Production**: Export to ONNX + TensorRT (additional 2-3x speedup)
   ```bash
   python dlshogi/convert_model_to_onnx.py model.ckpt model.onnx
   # Then optimize with TensorRT (FP16)
   ```

3. **Total speedup**: ~3-4x vs baseline PyTorch

### torch.compile + channels_last

Minimal additional benefit (~0.01x):

```python
model = model.to(memory_format=torch.channels_last)
model = torch.compile(model, mode="reduce-overhead")
```

Not recommended unless you need every bit of performance.

## Troubleshooting

### Compilation Warnings

```
UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.
```

Safe to ignore. Or enable for additional speedup:
```python
torch.set_float32_matmul_precision('high')
```

### Graph Breaks

If you see many graph breaks in profiling, the model may not be fully optimized. This is expected for models with:
- Dynamic control flow
- LayerNorm with permute operations
- Python loops

The InceptionNeXt architecture (exp015) has some graph breaks but still achieves 1.2-1.3x speedup.

### Compilation Time

First inference is slow due to compilation:
```python
# Warm up to trigger compilation
for _ in range(5):
    model(x1, x2)

# Now fast
model(x1, x2)  # Compiled version
```

## References

- exp017 documentation: `dlshogi/experiments/exp017_torch_compile/README.md`
- Benchmark script: `dlshogi/experiments/exp017_torch_compile/benchmark_optimizations.py`
- PyTorch docs: https://pytorch.org/docs/stable/torch.compiler.html

## Summary

- **Speedup**: 1.22-1.30x (batch size dependent)
- **Accuracy**: No change
- **Usage**: Add `--compile` flag to profile.py or `torch.compile(model)` in code
- **Recommendation**: Use for all inference/evaluation tasks
- **Next step**: Export to ONNX + TensorRT for production (additional 2-3x speedup)
