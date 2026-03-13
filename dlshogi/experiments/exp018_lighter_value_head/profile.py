"""Quick profiling for exp018 (lighter value head).

Usage:
    python profile.py [--batch-size 128]
"""
import argparse
import torch
import time
from dlshogi.experiments.exp018_lighter_value_head.model import PolicyValueNetwork
from dlshogi.common import *

def profile_model(batch_size=128, num_warmup=10, num_iterations=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyValueNetwork().to(device).eval()
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
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
    
    print(f"\nBatch size: {batch_size}")
    print(f"Average time: {avg_time:.2f} ms")
    print(f"Throughput: {throughput:.1f} samples/sec")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    
    profile_model(batch_size=args.batch_size)
