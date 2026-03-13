"""Profile torch.compile and channels_last optimizations.

Compares three configurations:
1. Baseline (no optimization)
2. torch.compile only
3. torch.compile + channels_last

Usage:
    bash profile.sh
    # or directly:
    python profile.py [--batch-size 128] [--runs 100]
"""
import argparse
import os
import time

import torch

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
from dlshogi.experiments.exp017_torch_compile.model import PolicyValueNetwork

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--runs", type=int, default=100)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"Batch size: {args.batch_size}")
print(f"Runs: {args.runs}\n")

# Prepare input
x1 = torch.randn(args.batch_size, FEATURES1_NUM, 9, 9, device=DEVICE)
x2 = torch.randn(args.batch_size, FEATURES2_NUM, 9, 9, device=DEVICE)


def benchmark(model, name, warmup=10):
    """Benchmark model inference time."""
    # Warmup
    for _ in range(warmup):
        model(x1, x2)
    
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(args.runs):
        model(x1, x2)
    
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    avg_time = elapsed / args.runs * 1000  # ms
    throughput = args.batch_size * args.runs / elapsed
    
    print(f"{name:40s} | {avg_time:7.3f} ms/batch | {throughput:8.1f} samples/s")
    return avg_time


print("=" * 80)
print("Configuration 1: Baseline (no optimization)")
print("=" * 80)
model_baseline = PolicyValueNetwork().to(DEVICE).eval()
time_baseline = benchmark(model_baseline, "Baseline")

print("\n" + "=" * 80)
print("Configuration 2: torch.compile only")
print("=" * 80)
model_compile = PolicyValueNetwork().to(DEVICE).eval()
model_compile = torch.compile(model_compile, mode="reduce-overhead")
time_compile = benchmark(model_compile, "torch.compile")

print("\n" + "=" * 80)
print("Configuration 3: torch.compile + channels_last")
print("=" * 80)
model_optimized = PolicyValueNetwork().to(DEVICE).eval()
model_optimized = model_optimized.to(memory_format=torch.channels_last)
model_optimized = torch.compile(model_optimized, mode="reduce-overhead")

# Convert inputs to channels_last
x1_cl = x1.to(memory_format=torch.channels_last)
x2_cl = x2.to(memory_format=torch.channels_last)

# Warmup
for _ in range(10):
    model_optimized(x1_cl, x2_cl)

if DEVICE == "cuda":
    torch.cuda.synchronize()

# Benchmark
start = time.perf_counter()
for _ in range(args.runs):
    model_optimized(x1_cl, x2_cl)

if DEVICE == "cuda":
    torch.cuda.synchronize()

elapsed = time.perf_counter() - start
time_optimized = elapsed / args.runs * 1000
throughput = args.batch_size * args.runs / elapsed

print(f"{'torch.compile + channels_last':40s} | {time_optimized:7.3f} ms/batch | {throughput:8.1f} samples/s")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print(f"Baseline:                    {time_baseline:7.3f} ms  (1.00x)")
print(f"torch.compile:               {time_compile:7.3f} ms  ({time_baseline/time_compile:.2f}x speedup)")
print(f"torch.compile + channels_last: {time_optimized:7.3f} ms  ({time_baseline/time_optimized:.2f}x speedup)")
