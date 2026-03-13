"""Comprehensive benchmark of torch.compile optimizations across batch sizes.

Tests baseline vs torch.compile vs torch.compile+channels_last
at batch sizes relevant for training (1024, 2048) and search (128, 256).

Usage:
    python benchmark_optimizations.py
"""
import time

import torch

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
from dlshogi.experiments.exp017_torch_compile.model import PolicyValueNetwork

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZES = [128, 256, 512, 1024, 2048]
WARMUP = 20
RUNS = 100

print(f"Device: {DEVICE}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name()}")
print()


def benchmark(model, x1, x2, batch_size, warmup=WARMUP, runs=RUNS):
    """Benchmark model inference time."""
    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            model(x1, x2)
    
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            model(x1, x2)
    
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    avg_time = elapsed / runs * 1000  # ms
    throughput = batch_size * runs / elapsed
    
    return avg_time, throughput


results = {}

for batch_size in BATCH_SIZES:
    print("=" * 80)
    print(f"Batch Size: {batch_size}")
    print("=" * 80)
    
    # Prepare inputs
    x1 = torch.randn(batch_size, FEATURES1_NUM, 9, 9, device=DEVICE)
    x2 = torch.randn(batch_size, FEATURES2_NUM, 9, 9, device=DEVICE)
    
    # 1. Baseline
    model_baseline = PolicyValueNetwork().to(DEVICE).eval()
    time_baseline, throughput_baseline = benchmark(model_baseline, x1, x2, batch_size)
    print(f"Baseline:                    {time_baseline:7.3f} ms | {throughput_baseline:8.1f} samples/s")
    
    # 2. torch.compile
    model_compile = PolicyValueNetwork().to(DEVICE).eval()
    model_compile = torch.compile(model_compile, mode="reduce-overhead")
    time_compile, throughput_compile = benchmark(model_compile, x1, x2, batch_size)
    speedup_compile = time_baseline / time_compile
    print(f"torch.compile:               {time_compile:7.3f} ms | {throughput_compile:8.1f} samples/s | {speedup_compile:.2f}x")
    
    # 3. torch.compile + channels_last
    model_optimized = PolicyValueNetwork().to(DEVICE).eval()
    model_optimized = model_optimized.to(memory_format=torch.channels_last)
    model_optimized = torch.compile(model_optimized, mode="reduce-overhead")
    
    x1_cl = x1.to(memory_format=torch.channels_last)
    x2_cl = x2.to(memory_format=torch.channels_last)
    
    time_optimized, throughput_optimized = benchmark(model_optimized, x1_cl, x2_cl, batch_size)
    speedup_optimized = time_baseline / time_optimized
    print(f"torch.compile + channels_last: {time_optimized:7.3f} ms | {throughput_optimized:8.1f} samples/s | {speedup_optimized:.2f}x")
    
    results[batch_size] = {
        'baseline': time_baseline,
        'compile': time_compile,
        'optimized': time_optimized,
        'speedup_compile': speedup_compile,
        'speedup_optimized': speedup_optimized,
    }
    
    print()

# Summary table
print("=" * 80)
print("Summary: Speedup vs Baseline")
print("=" * 80)
print(f"{'Batch Size':<12} | {'Baseline (ms)':<14} | {'torch.compile':<14} | {'compile+channels_last':<20}")
print("-" * 80)
for batch_size in BATCH_SIZES:
    r = results[batch_size]
    print(f"{batch_size:<12} | {r['baseline']:>12.3f} ms | {r['speedup_compile']:>12.2f}x | {r['speedup_optimized']:>18.2f}x")

print("\n" + "=" * 80)
print("Recommendations")
print("=" * 80)

# Find best configuration for each batch size
best_search = max(results[128]['speedup_compile'], results[128]['speedup_optimized'])
best_train = max(results[1024]['speedup_compile'], results[1024]['speedup_optimized'])

if best_search > 1.0:
    config = "torch.compile + channels_last" if results[128]['speedup_optimized'] > results[128]['speedup_compile'] else "torch.compile"
    print(f"✓ For search (batch=128): Use {config} ({best_search:.2f}x speedup)")
else:
    print(f"✗ For search (batch=128): Baseline is faster (no optimization needed)")

if best_train > 1.0:
    config = "torch.compile + channels_last" if results[1024]['speedup_optimized'] > results[1024]['speedup_compile'] else "torch.compile"
    print(f"✓ For training (batch=1024): Use {config} ({best_train:.2f}x speedup)")
else:
    print(f"✗ For training (batch=1024): Baseline is faster (no optimization needed)")

print("\nNote: torch.compile may show slowdown for small models due to:")
print("  - Compilation overhead not amortized over enough iterations")
print("  - Small model size makes baseline already memory-bound")
print("  - Graph breaks from dynamic operations (permute, LayerNorm)")
print("\nFor production deployment, consider TensorRT with FP16 for best performance.")
