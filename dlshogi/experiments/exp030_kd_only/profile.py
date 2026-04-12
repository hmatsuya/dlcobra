"""Profile this experiment with torch.profiler.

Usage:
    bash profile.sh
    # or directly:
    python profile.py [--batch-size 32] [--runs 20]
"""
import argparse
import os

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--runs", type=int, default=20)
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from dlshogi.experiments.exp022_plain_attention_tail.model import PolicyValueNetwork
print("Using exp022 model (InceptionNeXt + plain attention tail)")
model = PolicyValueNetwork().to(DEVICE).eval()

x1 = torch.zeros(args.batch_size, FEATURES1_NUM, 9, 9, device=DEVICE)
x2 = torch.zeros(args.batch_size, FEATURES2_NUM, 9, 9, device=DEVICE)

for _ in range(5):
    model(x1, x2)

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
