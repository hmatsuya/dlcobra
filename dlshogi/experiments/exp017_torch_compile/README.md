# exp017: PyTorch Compiler & Memory Layout Optimization

## Overview

Tests `torch.compile` and `channels_last` memory format optimizations on the InceptionNeXt architecture (exp015 baseline).

**Architecture**: depths=[10], dims=[192] (3.7M parameters)

## Optimization Strategies

1. **torch.compile**: JIT compilation with kernel fusion
2. **channels_last**: Native NHWC memory layout to reduce permute overhead

## Benchmark Results

Tested on NVIDIA GeForce RTX 3090 with PyTorch 2.10.0+cu128:

| Batch Size | Baseline (ms) | torch.compile | compile+channels_last |
|------------|---------------|---------------|-----------------------|
| 128        | 7.629         | 1.30x         | 1.30x                 |
| 256        | 14.599        | 1.24x         | 1.24x                 |
| 512        | 28.263        | 1.23x         | 1.24x                 |
| 1024       | 55.450        | 1.22x         | 1.22x                 |
| 2048       | 110.393       | 1.21x         | 1.22x                 |

## Key Findings

### Speedup Achieved
- **Search (batch=128)**: 1.30x speedup
- **Training (batch=1024)**: 1.22x speedup
- channels_last provides minimal additional benefit over torch.compile alone

### Why Not 2-3x?

The original optimization guide predicted 2-3x speedup, but actual results show ~1.2-1.3x because:

1. **Small model size**: 3.7M parameters means memory bandwidth isn't the bottleneck
2. **Graph breaks**: LayerNorm with permute operations prevent full kernel fusion
3. **Already optimized baseline**: PyTorch's eager mode is well-optimized for small models
4. **Limited permute overhead**: Only 2 permutes per block (stem + blocks)

### Recommendations

✓ **Use torch.compile for inference**: 1.2-1.3x speedup with zero accuracy loss
✓ **channels_last optional**: Minimal additional benefit (~0.01x)
✗ **Not recommended for training**: Compilation overhead may slow down first epoch

### Production Deployment

For maximum inference speed in the USI engine:
1. Export to ONNX: `python dlshogi/convert_model_to_onnx.py`
2. Optimize with TensorRT FP16: Expected 2-3x additional speedup
3. Total expected speedup: ~3-4x vs baseline PyTorch

## Usage

```bash
# Quick benchmark (batch=128, 100 runs)
bash benchmark.sh

# Comprehensive benchmark (all batch sizes)
python benchmark_optimizations.py

# Debug mode (verify model loads)
bash run.sh --debug
```

## Implementation Notes

The model architecture is identical to exp015 - only inference optimizations are applied:

```python
model = PolicyValueNetwork().to(device).eval()
model = model.to(memory_format=torch.channels_last)
model = torch.compile(model, mode="reduce-overhead")
```

No training is performed for this experiment since accuracy is unchanged.
