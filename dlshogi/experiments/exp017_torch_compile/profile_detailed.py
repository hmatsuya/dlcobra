"""Detailed profiling of exp015 InceptionNeXt model.

Analyzes:
- Layer-by-layer timing
- Memory usage
- CUDA kernel efficiency
- FLOPs breakdown
- Bottleneck identification
"""
import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import time
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.exp015_inceptionnext_depth10.model import PolicyValueNetwork
from dlshogi.common import *


def count_parameters(model):
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model, x1, x2):
    """Estimate FLOPs using dummy forward pass."""
    from torch.utils.flop_counter import FlopCounterMode
    
    flop_counter = FlopCounterMode(display=False)
    with flop_counter:
        _ = model(x1, x2)
    
    total_flops = flop_counter.get_total_flops()
    return total_flops


def profile_inference(model, x1, x2, num_warmup=10, num_iterations=100):
    """Profile inference time with warmup."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x1, x2)
    
    # Synchronize CUDA
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
    
    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
        'p95': np.percentile(times, 95),
        'p99': np.percentile(times, 99),
    }


def profile_with_pytorch_profiler(model, x1, x2):
    """Detailed profiling with PyTorch profiler."""
    model.eval()
    
    with torch.no_grad():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
        ) as prof:
            with record_function("model_inference"):
                _ = model(x1, x2)
    
    return prof


def analyze_layer_timing(prof):
    """Extract per-layer timing information."""
    print("\n" + "="*80)
    print("TOP 20 OPERATIONS BY CUDA TIME")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20,
        max_src_column_width=60
    ))
    
    print("\n" + "="*80)
    print("TOP 20 OPERATIONS BY CPU TIME")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cpu_time_total",
        row_limit=20,
        max_src_column_width=60
    ))
    
    print("\n" + "="*80)
    print("TOP 20 MEMORY CONSUMERS")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cuda_memory_usage",
        row_limit=20,
        max_src_column_width=60
    ))


def analyze_operation_breakdown(prof):
    """Breakdown by operation type."""
    events = prof.key_averages()
    
    # Categorize operations
    categories = {
        'conv': [],
        'linear': [],
        'norm': [],
        'activation': [],
        'memory': [],
        'other': []
    }
    
    for evt in events:
        name = evt.key.lower()
        # Use device_time_total for CUDA time (works across PyTorch versions)
        cuda_time = evt.device_time_total if hasattr(evt, 'device_time_total') else evt.cuda_time
        
        if 'conv' in name or 'depthwise' in name:
            categories['conv'].append(cuda_time)
        elif 'linear' in name or 'addmm' in name or 'matmul' in name:
            categories['linear'].append(cuda_time)
        elif 'norm' in name or 'batch_norm' in name or 'layer_norm' in name:
            categories['norm'].append(cuda_time)
        elif 'gelu' in name or 'relu' in name or 'sigmoid' in name or 'activation' in name:
            categories['activation'].append(cuda_time)
        elif 'copy' in name or 'permute' in name or 'view' in name or 'cat' in name:
            categories['memory'].append(cuda_time)
        else:
            categories['other'].append(cuda_time)
    
    print("\n" + "="*80)
    print("OPERATION BREAKDOWN BY CATEGORY")
    print("="*80)
    
    total_time = sum(sum(times) for times in categories.values())
    
    for category, times in categories.items():
        if times:
            cat_total = sum(times)
            cat_count = len(times)
            cat_avg = cat_total / cat_count if cat_count > 0 else 0
            percentage = (cat_total / total_time * 100) if total_time > 0 else 0
            
            print(f"{category.upper():15s}: {cat_total/1000:8.2f}ms ({percentage:5.1f}%) "
                  f"[{cat_count:3d} ops, avg: {cat_avg/1000:.2f}ms]")
    
    print(f"{'TOTAL':15s}: {total_time/1000:8.2f}ms (100.0%)")


def profile_memory_usage(model, x1, x2):
    """Profile memory usage."""
    if not torch.cuda.is_available():
        print("\nCUDA not available, skipping memory profiling")
        return
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
    # Measure model parameters memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    
    # Measure inference memory
    torch.cuda.synchronize()
    mem_before = torch.cuda.memory_allocated()
    
    with torch.no_grad():
        output = model(x1, x2)
    
    torch.cuda.synchronize()
    mem_after = torch.cuda.memory_allocated()
    mem_peak = torch.cuda.max_memory_allocated()
    
    print("\n" + "="*80)
    print("MEMORY USAGE")
    print("="*80)
    print(f"Model parameters:        {param_memory / 1024**2:8.2f} MB")
    print(f"Input tensors:           {(x1.numel() * x1.element_size() + x2.numel() * x2.element_size()) / 1024**2:8.2f} MB")
    print(f"Memory before inference: {mem_before / 1024**2:8.2f} MB")
    print(f"Memory after inference:  {mem_after / 1024**2:8.2f} MB")
    print(f"Peak memory:             {mem_peak / 1024**2:8.2f} MB")
    print(f"Activation memory:       {(mem_peak - mem_before) / 1024**2:8.2f} MB")


def profile_block_by_block(model, x1, x2):
    """Profile each InceptionNeXt block individually."""
    print("\n" + "="*80)
    print("PER-BLOCK TIMING (10 blocks)")
    print("="*80)
    
    model.eval()
    
    # Stem
    with torch.no_grad():
        x = model.stem_1(x1) + model.stem_2(x2)
        x = x.permute(0, 2, 3, 1)
        x = model.stem_norm(x)
        x = x.permute(0, 3, 1, 2)
    
    # Profile each block
    block_times = []
    for i, block in enumerate(model.blocks):
        times = []
        with torch.no_grad():
            for _ in range(50):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                x_out = block(x)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end = time.perf_counter()
                times.append((end - start) * 1000)
        
        x = x_out  # Update for next block
        avg_time = np.mean(times)
        block_times.append(avg_time)
        print(f"Block {i:2d}: {avg_time:6.3f}ms (±{np.std(times):.3f}ms)")
    
    print(f"{'Total blocks':10s}: {sum(block_times):6.3f}ms")
    
    # Profile heads
    with torch.no_grad():
        # Policy head
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        h_policy = model.policy(x)
        h_policy = torch.flatten(h_policy, 1) + model.policy_bias
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        policy_time = (time.perf_counter() - start) * 1000
        
        # Value head
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        h_value = model.value_conv(x)
        h_value = h_value.permute(0, 2, 3, 1)
        h_value = model.act(model.value_norm(h_value))
        h_value = h_value.permute(0, 3, 1, 2)
        h_value = model.act(model.value_fc1(torch.flatten(h_value, 1)))
        h_value = model.value_fc2(h_value)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        value_time = (time.perf_counter() - start) * 1000
    
    print(f"{'Policy head':10s}: {policy_time:6.3f}ms")
    print(f"{'Value head':10s}: {value_time:6.3f}ms")


def main():
    print("="*80)
    print("DETAILED PROFILING: exp015 InceptionNeXt (depth=10, dim=192)")
    print("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    
    # Create model
    model = PolicyValueNetwork(depths=[10], dims=[192], fcl=256)
    model = model.to(device)
    model.eval()
    
    # Create dummy inputs (batch_size=1 for inference)
    batch_size = 1
    x1 = torch.randn(batch_size, FEATURES1_NUM, 9, 9, device=device)
    x2 = torch.randn(batch_size, FEATURES2_NUM, 9, 9, device=device)
    
    # Model info
    total_params, trainable_params = count_parameters(model)
    print(f"\nTotal parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
    
    # Estimate FLOPs
    try:
        flops = estimate_flops(model, x1, x2)
        print(f"FLOPs per inference: {flops:,} ({flops/1e9:.2f} GFLOPs)")
    except Exception as e:
        print(f"Could not estimate FLOPs: {e}")
    
    # Basic inference timing
    print("\n" + "="*80)
    print("INFERENCE TIMING (100 iterations)")
    print("="*80)
    timing_stats = profile_inference(model, x1, x2, num_warmup=10, num_iterations=100)
    print(f"Mean:   {timing_stats['mean']:.3f}ms")
    print(f"Median: {timing_stats['median']:.3f}ms")
    print(f"Std:    {timing_stats['std']:.3f}ms")
    print(f"Min:    {timing_stats['min']:.3f}ms")
    print(f"Max:    {timing_stats['max']:.3f}ms")
    print(f"P95:    {timing_stats['p95']:.3f}ms")
    print(f"P99:    {timing_stats['p99']:.3f}ms")
    
    # Memory profiling
    profile_memory_usage(model, x1, x2)
    
    # Block-by-block timing
    profile_block_by_block(model, x1, x2)
    
    # Detailed PyTorch profiler
    print("\n" + "="*80)
    print("RUNNING PYTORCH PROFILER...")
    print("="*80)
    prof = profile_with_pytorch_profiler(model, x1, x2)
    
    # Analyze results
    analyze_layer_timing(prof)
    analyze_operation_breakdown(prof)
    
    # Export trace for visualization
    trace_file = Path(__file__).parent / "profile_trace.json"
    prof.export_chrome_trace(str(trace_file))
    print(f"\n{'='*80}")
    print(f"Chrome trace exported to: {trace_file}")
    print(f"View at: chrome://tracing")
    print(f"{'='*80}")
    
    # Test with realistic batch sizes (training=1024, search=128)
    print("\n" + "="*80)
    print("BATCH SIZE COMPARISON (Training & Search)")
    print("="*80)
    print(f"{'Batch':>6s} | {'Total (ms)':>10s} | {'Per Sample (ms)':>16s} | {'Throughput':>15s} | {'Use Case':>20s}")
    print("-"*80)
    
    batch_configs = [
        (1, "Single position"),
        (16, "Small batch"),
        (32, "MCTS (small)"),
        (64, "MCTS (medium)"),
        (128, "MCTS/Search (default)"),
        (256, "Training (small)"),
        (512, "Training (medium)"),
        (1024, "Training (default)"),
    ]
    
    for bs, use_case in batch_configs:
        try:
            x1_batch = torch.randn(bs, FEATURES1_NUM, 9, 9, device=device)
            x2_batch = torch.randn(bs, FEATURES2_NUM, 9, 9, device=device)
            
            stats = profile_inference(model, x1_batch, x2_batch, num_warmup=5, num_iterations=50)
            per_sample = stats['mean'] / bs
            throughput = bs / (stats['mean'] / 1000)  # positions/second
            
            print(f"{bs:6d} | {stats['mean']:10.3f} | {per_sample:16.6f} | {throughput:10.1f} pos/s | {use_case:>20s}")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"{bs:6d} | OUT OF MEMORY")
                break
            else:
                raise
    
    # Detailed analysis for training and search batch sizes
    print("\n" + "="*80)
    print("DETAILED ANALYSIS: TRAINING (batch=1024) vs SEARCH (batch=128)")
    print("="*80)
    
    # Training batch size
    print("\nTraining Configuration (batch=1024):")
    x1_train = torch.randn(1024, FEATURES1_NUM, 9, 9, device=device)
    x2_train = torch.randn(1024, FEATURES2_NUM, 9, 9, device=device)
    train_stats = profile_inference(model, x1_train, x2_train, num_warmup=10, num_iterations=100)
    print(f"  Total time per batch:  {train_stats['mean']:.3f}ms (±{train_stats['std']:.3f}ms)")
    print(f"  Time per sample:       {train_stats['mean']/1024:.6f}ms")
    print(f"  Throughput:            {1024/(train_stats['mean']/1000):.1f} positions/second")
    print(f"  Batches per second:    {1000/train_stats['mean']:.2f}")
    
    # Search batch size
    print("\nSearch Configuration (batch=128):")
    x1_search = torch.randn(128, FEATURES1_NUM, 9, 9, device=device)
    x2_search = torch.randn(128, FEATURES2_NUM, 9, 9, device=device)
    search_stats = profile_inference(model, x1_search, x2_search, num_warmup=10, num_iterations=100)
    print(f"  Total time per batch:  {search_stats['mean']:.3f}ms (±{search_stats['std']:.3f}ms)")
    print(f"  Time per sample:       {search_stats['mean']/128:.6f}ms")
    print(f"  Throughput:            {128/(search_stats['mean']/1000):.1f} positions/second")
    print(f"  Batches per second:    {1000/search_stats['mean']:.2f}")
    
    # MCTS implications
    print("\n" + "="*80)
    print("MCTS SEARCH PERFORMANCE (batch=128)")
    print("="*80)
    
    time_per_sample = search_stats['mean'] / 128
    
    print("\nPositions evaluated per time budget:")
    for time_budget_ms in [10, 50, 100, 200, 500, 1000]:
        positions = int(time_budget_ms / time_per_sample)
        batches = int(np.ceil(positions / 128))
        actual_time = batches * search_stats['mean']
        actual_positions = batches * 128
        print(f"  {time_budget_ms:4d}ms budget: ~{actual_positions:5d} positions ({batches:3d} batches, {actual_time:.1f}ms actual)")
    
    print("\nComparison with single-position inference:")
    single_time = 3.063  # From earlier profiling
    print(f"  Single position:  {single_time:.3f}ms")
    print(f"  Batch=128:        {time_per_sample:.6f}ms per position")
    print(f"  Speedup:          {single_time/time_per_sample:.1f}x faster with batching")


if __name__ == "__main__":
    main()
