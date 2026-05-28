"""Quick profiling for exp035 (batch=128 default)."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

import torch
from dlshogi.experiments.exp026_droppath_bf16_fresh.model import PolicyValueNetwork
from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM

batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = PolicyValueNetwork().to(device).eval()
x1 = torch.randn(batch_size, FEATURES1_NUM, 9, 9, device=device)
x2 = torch.randn(batch_size, FEATURES2_NUM, 9, 9, device=device)

# Warmup
for _ in range(10):
    with torch.no_grad():
        model(x1, x2)

import time
torch.cuda.synchronize()
t0 = time.perf_counter()
N = 100
for _ in range(N):
    with torch.no_grad():
        model(x1, x2)
torch.cuda.synchronize()
elapsed = time.perf_counter() - t0
print(f"batch={batch_size}, {N} iters, {elapsed:.3f}s, {elapsed/N*1000:.2f}ms/iter, {batch_size*N/elapsed:.0f} pos/s")
