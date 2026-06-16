"""Find a hybrid config whose inference speed matches InceptionNeXt.

At equal params the hybrid is ~1.5x faster than InceptionNeXt, so a
speed-matched hybrid can be larger (more capacity). We scale the hybrid
(channels / blocks) until its pos/s matches InceptionNeXt at the target batch,
and report the resulting param count.
"""
import argparse
import sys

import torch

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
from dlshogi.experiments.exp026_droppath_bf16_fresh.model import PolicyValueNetwork as INX
from dlshogi.experiments.exp037_trt_backbone_benchmark.hybrid_model import PolicyValueNetwork as HYBRID


@torch.no_grad()
def time_model(model, x1, x2, iters, warmup):
    for _ in range(warmup):
        model(x1, x2)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        model(x1, x2)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def bench(model, batch, iters, warmup, compile_mode):
    device = torch.device("cuda")
    model = model.to(device).eval().half()
    n_params = sum(p.numel() for p in model.parameters())
    x1 = torch.randn(batch, FEATURES1_NUM, 9, 9, device=device, dtype=torch.float16)
    x2 = torch.randn(batch, FEATURES2_NUM, 9, 9, device=device, dtype=torch.float16)
    if compile_mode:
        model = torch.compile(model, mode="reduce-overhead")
    ms = time_model(model, x1, x2, iters, warmup)
    torch.cuda.empty_cache()
    if compile_mode:
        torch._dynamo.reset()
    return n_params, ms, batch * 1000.0 / ms


def main(*argv):
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=80)
    p.add_argument("--compile", action="store_true")
    p.add_argument("--period", type=int, default=5)
    # candidate hybrid (channels, blocks) configs to sweep
    p.add_argument("--configs", nargs="+",
                   default=["384:36", "448:36", "448:40", "480:40", "512:36", "512:40", "512:44"],
                   help="list of channels:blocks")
    args = p.parse_args(argv)

    print(f"# GPU {torch.cuda.get_device_name(0)} | batch={args.batch} compile={args.compile} period={args.period}")

    # baseline
    n_inx, ms_inx, pos_inx = bench(INX(), args.batch, args.iters, args.warmup, args.compile)
    print(f"\nBASELINE inceptionnext: {n_inx/1e6:.1f}M  {ms_inx:.3f} ms  {pos_inx:.1f} pos/s\n")

    print(f"{'hybrid cfg':<16} | {'params':>8} | {'mean ms':>9} | {'pos/s':>9} | {'vs INX spd':>10}")
    print("-" * 64)
    for cfg in args.configs:
        ch, bl = (int(v) for v in cfg.split(":"))
        # channels must be divisible by 3 (axial) and 4-friendly; round to mult of 12
        if ch % 3 != 0:
            print(f"{cfg:<16} | skipped (channels%3!=0)")
            continue
        try:
            model = HYBRID(blocks=bl, channels=ch, axial_period=args.period)
            n, ms, pos = bench(model, args.batch, args.iters, args.warmup, args.compile)
            ratio = pos / pos_inx
            flag = "  <== match" if 0.95 <= ratio <= 1.05 else ""
            print(f"{f'ch{ch}x{bl}':<16} | {n/1e6:>7.1f}M | {ms:>9.4f} | {pos:>9.1f} | {ratio:>9.2f}x{flag}")
        except Exception as e:
            print(f"{cfg:<16} | ERROR {e}", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:])
