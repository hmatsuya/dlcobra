"""GPU FP16 inference latency benchmark: InceptionNeXt vs equal-param ResNet/SE-ResNet.

TensorRT (trtexec) is blocked on this machine by a driver/runtime mismatch, so we
measure on the next-best, fully-functional path: PyTorch FP16 on GPU, both eager
and torch.compile (the exp017 production path). The relative ranking is what we
care about: depthwise convs (InceptionNeXt) are memory-bound and underutilize
tensor cores, while ResNet/SE-ResNet dense 3x3 convs are math-bound and map onto
tensor cores -- this shows up on the CUDA backend regardless of TensorRT.

Networks (all ~86M params):
  inceptionnext   86.14M  (depthwise 3x3 + 1x9 + 9x1, exp026 arch)
  resnet32x384    85.83M  (dense 3x3)
  senet32x384     87.02M  (dense 3x3 + SE)
"""
import argparse
import sys
import time

import torch

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM

NETWORKS = ["inceptionnext", "resnet32x384", "senet32x384", "hybrid36p5", "hybrid37p4", "hybrid_exp038"]

# hybrid variants: (blocks, channels, axial_period)
HYBRID_CFG = {
    "hybrid36p5": (36, 384, 5),   # 86.15M: 29 dense + 7 axial (equal-param)
    "hybrid37p4": (37, 384, 4),   # 85.87M: 28 dense + 9 axial (equal-param)
    "hybrid_exp038": (36, 480, 5), # 134.2M: 29 dense + 7 axial (speed-matched)
}


def build_model(network):
    if network == "inceptionnext":
        from dlshogi.experiments.exp026_droppath_bf16_fresh.model import PolicyValueNetwork
        return PolicyValueNetwork()
    if network in HYBRID_CFG:
        from dlshogi.experiments.exp037_trt_backbone_benchmark.hybrid_model import PolicyValueNetwork
        blocks, channels, period = HYBRID_CFG[network]
        return PolicyValueNetwork(blocks=blocks, channels=channels, axial_period=period)
    from dlshogi.network.policy_value_network import policy_value_network
    return policy_value_network(network, add_sigmoid=True)


@torch.no_grad()
def time_model(model, x1, x2, iters, warmup):
    # warmup
    for _ in range(warmup):
        model(x1, x2)
    torch.cuda.synchronize()
    # timed
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        model(x1, x2)
    end.record()
    torch.cuda.synchronize()
    total_ms = start.elapsed_time(end)
    return total_ms / iters  # ms per inference


def run(network, batch, iters, warmup, compile_mode):
    device = torch.device("cuda")
    model = build_model(network).to(device).eval().half()
    n_params = sum(p.numel() for p in model.parameters())

    x1 = torch.randn(batch, FEATURES1_NUM, 9, 9, device=device, dtype=torch.float16)
    x2 = torch.randn(batch, FEATURES2_NUM, 9, 9, device=device, dtype=torch.float16)

    if compile_mode:
        model = torch.compile(model, mode="reduce-overhead")

    mean_ms = time_model(model, x1, x2, iters, warmup)
    pos_s = batch * 1000.0 / mean_ms
    return n_params, mean_ms, pos_s


def main(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 256])
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--compile", action="store_true", help="use torch.compile (reduce-overhead)")
    args = parser.parse_args(argv)

    print(f"# GPU: {torch.cuda.get_device_name(0)}")
    print(f"# torch {torch.__version__}, FP16, compile={args.compile}")
    print(f"# iters={args.iters} warmup={args.warmup}")
    print()
    header = f"{'batch':>6} | {'network':<16} | {'params':>8} | {'mean ms':>9} | {'pos/s':>11} | {'vs INX':>7}"
    print(header)
    print("-" * len(header))

    for bs in args.batches:
        results = {}
        for net in NETWORKS:
            try:
                results[net] = run(net, bs, args.iters, args.warmup, args.compile)
            except Exception as e:
                print(f"{bs:>6} | {net:<16} | ERROR: {e}", file=sys.stderr)
                results[net] = None
            torch.cuda.empty_cache()
            if args.compile:
                torch._dynamo.reset()
        inx = results.get("inceptionnext")
        inx_pos = inx[2] if inx else None
        for net in NETWORKS:
            r = results.get(net)
            if r is None:
                print(f"{bs:>6} | {net:<16} | {'FAIL':>8}")
                continue
            n_params, mean_ms, pos_s = r
            speedup = (pos_s / inx_pos) if inx_pos else float("nan")
            print(f"{bs:>6} | {net:<16} | {n_params/1e6:>7.1f}M | {mean_ms:>9.4f} | {pos_s:>11.1f} | {speedup:>6.2f}x")
        print("-" * len(header))

    print("\nNotes:")
    print("- mean ms = wall GPU time per forward call (lower better).")
    print("- pos/s   = batch / mean_ms = positions/sec (higher better).")
    print("- vs INX  = pos/s relative to InceptionNeXt at same batch (>1 = faster).")


if __name__ == "__main__":
    main(*sys.argv[1:])
