"""Profile exp014 Hierarchical InceptionNeXt with torch.profiler."""
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from dlshogi.experiments.exp014_inceptionnext_hierarchical.model import PolicyValueNetwork

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = PolicyValueNetwork().to(DEVICE).eval()
x1 = torch.zeros(BATCH_SIZE, 62, 9, 9, device=DEVICE)
x2 = torch.zeros(BATCH_SIZE, 57, 9, 9, device=DEVICE)

# Warm up
for _ in range(5):
    model(x1, x2)

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_flops=True,
    profile_memory=True,
) as prof:
    with record_function("forward"):
        for _ in range(20):
            model(x1, x2)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
prof.export_chrome_trace("trace_exp014.json")
print("\nChrome trace saved to trace_exp014.json")
print("Open at https://ui.perfetto.dev")
