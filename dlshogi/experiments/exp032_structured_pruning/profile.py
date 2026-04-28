"""Profile exp032 pruned model and compare throughput against exp029 (unpruned).

Usage:
    bash profile.sh
    python profile.py [--batch-size 128] [--runs 20] [--compile] [--compare]

Options:
    --compile:  Apply torch.compile (reduce-overhead mode)
    --compare:  Also benchmark the original unpruned model for comparison
"""
import argparse
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function
import os, sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
from dlshogi.experiments.exp032_structured_pruning.pruned_network import PrunedPolicyValueNetwork, PRUNED_MLP_DIM

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--runs", type=int, default=20)
parser.add_argument("--compile", action="store_true")
parser.add_argument("--compare", action="store_true", help="Also benchmark unpruned exp026 model")
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def benchmark(model, x1, x2, runs: int, label: str) -> float:
    model.eval()
    # Warm up
    for _ in range(5):
        with torch.no_grad():
            model(x1, x2)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        with torch.no_grad():
            model(x1, x2)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    ms = elapsed / runs * 1000
    tput = args.batch_size * runs / elapsed
    print(f"  {label:40s}  {ms:7.2f} ms/batch  {tput:8.0f} samples/s")
    return ms


x1 = torch.zeros(args.batch_size, FEATURES1_NUM, 9, 9, device=DEVICE)
x2 = torch.zeros(args.batch_size, FEATURES2_NUM, 9, 9, device=DEVICE)

print(f"\n=== exp032 Pruned Model Benchmark (batch={args.batch_size}, device={DEVICE}) ===")
print(f"mlp_expansion_dim: {PRUNED_MLP_DIM}  (original: 2048)")

pruned = PrunedPolicyValueNetwork().to(DEVICE)
n_params = sum(p.numel() for p in pruned.parameters())
print(f"Parameters: {n_params/1e6:.2f}M\n")

if args.compile:
    print("Applying torch.compile...")
    pruned = torch.compile(pruned, mode="reduce-overhead")

ms_pruned = benchmark(pruned, x1, x2, args.runs, f"exp032 pruned (dim={PRUNED_MLP_DIM})")

if args.compare:
    from dlshogi.experiments.exp026_droppath_bf16_fresh.model import PolicyValueNetwork as OrigNet
    orig = OrigNet().to(DEVICE)
    n_orig = sum(p.numel() for p in orig.parameters())
    print(f"\nOriginal (exp026/exp029): {n_orig/1e6:.2f}M params")
    if args.compile:
        orig = torch.compile(orig, mode="reduce-overhead")
    ms_orig = benchmark(orig, x1, x2, args.runs, "exp026/exp029 original (dim=2048)")
    speedup = ms_orig / ms_pruned
    param_reduction = (n_orig - n_params) / n_orig
    print(f"\n  Speedup:           {speedup:.2f}x")
    print(f"  Param reduction:   {param_reduction:.1%}")

# Detailed profiling of pruned model
print("\n=== Detailed profiling (pruned model) ===")
pruned_for_profile = PrunedPolicyValueNetwork().to(DEVICE).eval()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
    profile_memory=True,
) as prof:
    with record_function("forward"):
        for _ in range(args.runs):
            with torch.no_grad():
                pruned_for_profile(x1, x2)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

trace_path = os.path.join(_SCRIPT_DIR, "trace.json")
prof.export_chrome_trace(trace_path)
print(f"\nChrome trace → {trace_path}")
print("View at https://ui.perfetto.dev")
