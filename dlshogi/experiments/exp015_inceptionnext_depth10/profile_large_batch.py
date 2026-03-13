"""Profile exp015 with large batch sizes for throughput analysis.

Tests batch sizes: 64, 128, 256, 512, 1024, 2048, 4096
Measures throughput (positions/second) and memory usage.
"""
import torch
import torch.nn as nn
import time
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.exp015_inceptionnext_depth10.model import PolicyValueNetwork
from dlshogi.common import *


def profile_batch_size(model, batch_size, num_warmup=5, num_iterations=20, device='cuda'):
    """Profile inference with specific batch size."""
    model.eval()
    
    # Create inputs
    x1 = torch.randn(batch_size, FEATURES1_NUM, 9, 9, device=device)
    x2 = torch.randn(batch_size, FEATURES2_NUM, 9, 9, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x1, x2)
    
    # Synchronize
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.perf_counter()
            _ = model(x1, x2)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms
    
    # Memory usage
    if torch.cuda.is_available():
        mem_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        mem_reserved = torch.cuda.memory_reserved() / 1024**2    # MB
    else:
        mem_allocated = 0
        mem_reserved = 0
    
    return {
        'batch_size': batch_size,
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'min_time': np.min(times),
        'max_time': np.max(times),
        'median_time': np.median(times),
        'per_sample_time': np.mean(times) / batch_size,
        'throughput': batch_size / (np.mean(times) / 1000),  # positions/second
        'mem_allocated': mem_allocated,
        'mem_reserved': mem_reserved,
    }


def main():
    print("="*80)
    print("LARGE BATCH SIZE PROFILING: exp015 InceptionNeXt")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total GPU Memory: {total_mem:.2f} GB")
    
    # Create model
    model = PolicyValueNetwork(depths=[10], dims=[192], fcl=256)
    model = model.to(device)
    model.eval()
    
    # Test batch sizes
    batch_sizes = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    
    print("\n" + "="*80)
    print("BATCH SIZE SCALING ANALYSIS")
    print("="*80)
    print(f"\n{'Batch':>6s} | {'Total (ms)':>10s} | {'Per Sample (ms)':>16s} | "
          f"{'Throughput':>12s} | {'Memory (MB)':>12s} | {'Speedup':>8s}")
    print("-"*80)
    
    results = []
    baseline_per_sample = None
    
    for batch_size in batch_sizes:
        try:
            # Clear cache before each test
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            
            result = profile_batch_size(model, batch_size, num_warmup=5, num_iterations=20, device=device)
            results.append(result)
            
            if baseline_per_sample is None:
                baseline_per_sample = result['per_sample_time']
            
            speedup = baseline_per_sample / result['per_sample_time']
            
            print(f"{batch_size:6d} | {result['mean_time']:10.3f} | "
                  f"{result['per_sample_time']:16.6f} | "
                  f"{result['throughput']:9.1f} pos/s | "
                  f"{result['mem_allocated']:12.2f} | "
                  f"{speedup:8.2f}x")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{batch_size:6d} | OUT OF MEMORY")
                break
            else:
                raise
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    if len(results) >= 2:
        # Find optimal batch size (best throughput)
        best_result = max(results, key=lambda x: x['throughput'])
        print(f"\nOptimal Batch Size: {best_result['batch_size']}")
        print(f"  Throughput: {best_result['throughput']:.1f} positions/second")
        print(f"  Per-sample time: {best_result['per_sample_time']:.6f} ms")
        print(f"  Total time: {best_result['mean_time']:.3f} ms")
        print(f"  Memory: {best_result['mem_allocated']:.2f} MB")
        
        # Scaling efficiency
        print("\nScaling Efficiency:")
        for i, result in enumerate(results):
            if i == 0:
                continue
            expected_speedup = result['batch_size']
            actual_speedup = baseline_per_sample / result['per_sample_time']
            efficiency = (actual_speedup / expected_speedup) * 100
            print(f"  Batch {result['batch_size']:4d}: {efficiency:5.1f}% efficient "
                  f"({actual_speedup:.1f}x vs {expected_speedup}x ideal)")
        
        # Memory scaling
        print("\nMemory Scaling:")
        for result in results:
            mem_per_sample = result['mem_allocated'] / result['batch_size']
            print(f"  Batch {result['batch_size']:4d}: {result['mem_allocated']:7.2f} MB total, "
                  f"{mem_per_sample:.4f} MB per sample")
        
        # Throughput comparison
        print("\nThroughput Comparison:")
        print(f"  Batch=1:    {results[0]['throughput']:8.1f} pos/s (baseline)")
        if len(results) > 1:
            print(f"  Batch=32:   {results[5]['throughput']:8.1f} pos/s "
                  f"({results[5]['throughput']/results[0]['throughput']:.1f}x)")
        if len(results) > 6:
            print(f"  Batch=128:  {results[7]['throughput']:8.1f} pos/s "
                  f"({results[7]['throughput']/results[0]['throughput']:.1f}x)")
        if len(results) > 9:
            print(f"  Batch=1024: {results[10]['throughput']:8.1f} pos/s "
                  f"({results[10]['throughput']/results[0]['throughput']:.1f}x)")
        print(f"  Best:       {best_result['throughput']:8.1f} pos/s "
              f"(batch={best_result['batch_size']}, "
              f"{best_result['throughput']/results[0]['throughput']:.1f}x)")
    
    # MCTS implications
    print("\n" + "="*80)
    print("MCTS IMPLICATIONS")
    print("="*80)
    
    print("\nFor different time budgets per move:")
    time_budgets = [10, 50, 100, 200, 500, 1000]  # milliseconds
    
    for time_budget in time_budgets:
        print(f"\n  {time_budget}ms budget:")
        for result in results[:min(len(results), 8)]:  # Show first 8 batch sizes
            positions_per_move = int(time_budget / result['per_sample_time'])
            batches_needed = int(np.ceil(positions_per_move / result['batch_size']))
            actual_time = batches_needed * result['mean_time']
            actual_positions = batches_needed * result['batch_size']
            
            if actual_time <= time_budget * 1.1:  # Within 10% of budget
                print(f"    Batch {result['batch_size']:4d}: ~{actual_positions:5d} positions "
                      f"({batches_needed} batches, {actual_time:.1f}ms)")
    
    # Save results
    output_file = Path(__file__).parent / "large_batch_results.txt"
    with open(output_file, 'w') as f:
        f.write("Batch Size Profiling Results\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Batch':>6s} | {'Total (ms)':>10s} | {'Per Sample (ms)':>16s} | "
                f"{'Throughput':>12s} | {'Speedup':>8s}\n")
        f.write("-"*80 + "\n")
        
        for result in results:
            speedup = baseline_per_sample / result['per_sample_time']
            f.write(f"{result['batch_size']:6d} | {result['mean_time']:10.3f} | "
                    f"{result['per_sample_time']:16.6f} | "
                    f"{result['throughput']:9.1f} pos/s | "
                    f"{speedup:8.2f}x\n")
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
