"""Detailed profiling with PyTorch profiler for exp018.

Generates Chrome trace file for visualization.
"""
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from dlshogi.experiments.exp018_lighter_value_head.model import PolicyValueNetwork
from dlshogi.common import *

def profile_detailed(batch_size=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyValueNetwork().to(device).eval()
    
    x1 = torch.randn(batch_size, FEATURES1_NUM, 9, 9, device=device)
    x2 = torch.randn(batch_size, FEATURES2_NUM, 9, 9, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x1, x2)
    
    # Profile
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    
    with profile(
        activities=activities,
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with torch.no_grad():
            for _ in range(10):
                with record_function("model_inference"):
                    _ = model(x1, x2)
    
    # Print summary
    print(prof.key_averages().table(sort_by="cuda_time_total" if device.type == "cuda" else "cpu_time_total", row_limit=20))
    
    # Export trace
    prof.export_chrome_trace("dlshogi/experiments/exp018_lighter_value_head/trace.json")
    print("\nTrace exported to: dlshogi/experiments/exp018_lighter_value_head/trace.json")

if __name__ == "__main__":
    profile_detailed()
