"""Profile this experiment with torch.profiler.

Usage:
    bash profile.sh
    # or directly:
    python profile.py [--batch-size 128] [--runs 20] [--compile]
    
Options:
    --compile: Use torch.compile for 1.2-1.3x speedup (exp017 verified)
"""
import argparse
import importlib
import os
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=128)  # Default matches search batch size
parser.add_argument("--runs", type=int, default=20)
parser.add_argument("--compile", action="store_true", help="Use torch.compile (1.2-1.3x speedup)")
args = parser.parse_args()

# Resolve experiment module from this file's location
script_dir = os.path.dirname(os.path.abspath(__file__))
exp_name = os.path.basename(script_dir)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    module = importlib.import_module(f"dlshogi.experiments.{exp_name}.model")
    PolicyValueNetwork = module.PolicyValueNetwork
    print(f"Using custom model from {exp_name}.model")
    model = PolicyValueNetwork().to(DEVICE).eval()
except ModuleNotFoundError:
    from dlshogi.network.policy_value_network import policy_value_network
    print("No custom model found, using base network (resnet10_relu)")
    model = policy_value_network("resnet10_relu").to(DEVICE).eval()

if args.compile:
    print("Applying torch.compile (mode=reduce-overhead)...")
    model = torch.compile(model, mode="reduce-overhead")

x1 = torch.zeros(args.batch_size, FEATURES1_NUM, 9, 9, device=DEVICE)
x2 = torch.zeros(args.batch_size, FEATURES2_NUM, 9, 9, device=DEVICE)

# Warm up
print("Warming up...")
for _ in range(5):
    model(x1, x2)

# Quick benchmark
if DEVICE == "cuda":
    torch.cuda.synchronize()
start = time.perf_counter()
for _ in range(args.runs):
    with torch.no_grad():
        model(x1, x2)
if DEVICE == "cuda":
    torch.cuda.synchronize()
elapsed = time.perf_counter() - start
avg_time = elapsed / args.runs * 1000
throughput = args.batch_size * args.runs / elapsed

print(f"\nBenchmark: {avg_time:.3f} ms/batch | {throughput:.1f} samples/s")
if args.compile:
    print("Note: torch.compile provides 1.22-1.30x speedup (exp017)")

# Detailed profiling
print("\nProfiling...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
    profile_memory=True,
) as prof:
    with record_function("forward"):
        for _ in range(args.runs):
            model(x1, x2)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

trace_path = os.path.join(script_dir, "trace.json")
prof.export_chrome_trace(trace_path)
print(f"\nChrome trace saved to {trace_path}")
print("Open at https://ui.perfetto.dev")
