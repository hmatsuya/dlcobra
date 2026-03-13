"""Compare exp018 (lighter value head) with exp015 (baseline).

Measures inference time and parameter count for both models.
"""
import torch
import time
from dlshogi.experiments.exp015_inceptionnext_depth10.model import PolicyValueNetwork as Exp015Model
from dlshogi.experiments.exp018_lighter_value_head.model import PolicyValueNetwork as Exp018Model
from dlshogi.common import *

def benchmark_model(model, name, batch_size=128, num_warmup=10, num_iterations=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Create dummy inputs
    x1 = torch.randn(batch_size, FEATURES1_NUM, 9, 9, device=device)
    x2 = torch.randn(batch_size, FEATURES2_NUM, 9, 9, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x1, x2)
    
    # Benchmark
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    start = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(x1, x2)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    avg_time = elapsed / num_iterations * 1000  # ms
    throughput = batch_size * num_iterations / elapsed
    
    return {
        'name': name,
        'params': total_params,
        'avg_time_ms': avg_time,
        'throughput': throughput
    }

if __name__ == "__main__":
    print("=" * 70)
    print("Comparing exp015 (baseline) vs exp018 (lighter value head)")
    print("=" * 70)
    
    exp015 = benchmark_model(Exp015Model(), "exp015 (baseline)")
    exp018 = benchmark_model(Exp018Model(), "exp018 (lighter value head)")
    
    print(f"\n{'Model':<30} {'Params':<15} {'Time (ms)':<12} {'Throughput':<15}")
    print("-" * 70)
    print(f"{exp015['name']:<30} {exp015['params']:>14,} {exp015['avg_time_ms']:>11.2f} {exp015['throughput']:>14.1f}")
    print(f"{exp018['name']:<30} {exp018['params']:>14,} {exp018['avg_time_ms']:>11.2f} {exp018['throughput']:>14.1f}")
    
    param_reduction = exp015['params'] - exp018['params']
    param_reduction_pct = (param_reduction / exp015['params']) * 100
    speedup = exp015['avg_time_ms'] / exp018['avg_time_ms']
    
    print("-" * 70)
    print(f"Parameter reduction: {param_reduction:,} ({param_reduction_pct:.1f}%)")
    print(f"Speedup: {speedup:.2f}x ({((speedup - 1) * 100):.1f}% faster)")
    print("=" * 70)
