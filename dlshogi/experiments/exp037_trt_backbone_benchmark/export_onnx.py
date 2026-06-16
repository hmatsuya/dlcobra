"""Export a network to ONNX (random weights) for TensorRT latency benchmarking.

We only measure inference latency here, so weights are random (untrained).
Networks compared:
  - InceptionNeXt (exp026 architecture, 86.1M)
  - resnet32x384 (85.8M, equal-param)
  - senet32x384  (87.0M, equal-param)

Usage:
  python export_onnx.py <network> <out.onnx> --batch 256
"""
import argparse
import sys

import torch

from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM

# hybrid variants: (blocks, channels, axial_period)
HYBRID_CFG = {
    "hybrid36p5": (36, 384, 5),   # 86.15M: 29 dense + 7 axial (equal-param)
    "hybrid37p4": (37, 384, 4),   # 85.87M: 28 dense + 9 axial (equal-param)
    "hybrid_exp038": (36, 480, 5), # 134.2M: 29 dense + 7 axial (speed-matched)
}


def build_model(network):
    if network == "inceptionnext":
        from dlshogi.experiments.exp026_droppath_bf16_fresh.model import PolicyValueNetwork
        model = PolicyValueNetwork()
    elif network in HYBRID_CFG:
        from dlshogi.experiments.exp037_trt_backbone_benchmark.hybrid_model import PolicyValueNetwork
        blocks, channels, period = HYBRID_CFG[network]
        model = PolicyValueNetwork(blocks=blocks, channels=channels, axial_period=period)
    else:
        from dlshogi.network.policy_value_network import policy_value_network
        model = policy_value_network(network, add_sigmoid=True)
    return model


def main(*argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("network")
    parser.add_argument("onnx")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--fp16", action="store_true",
                        help="Export in FP16. This TRT build is strongly-typed, "
                             "so ONNX precision dictates engine precision.")
    args = parser.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.network).to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{args.network}] params = {n_params/1e6:.2f}M", file=sys.stderr)

    dtype = torch.float16 if args.fp16 else torch.float32
    if args.fp16:
        model = model.half()

    x1 = torch.randn(args.batch, FEATURES1_NUM, 9, 9, device=device, dtype=dtype)
    x2 = torch.randn(args.batch, FEATURES2_NUM, 9, 9, device=device, dtype=dtype)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (x1, x2),
            args.onnx,
            do_constant_folding=True,
            opset_version=args.opset,
            input_names=["input1", "input2"],
            output_names=["output_policy", "output_value"],
        )
    print(f"[{args.network}] exported -> {args.onnx} (fixed batch={args.batch}, fp16={args.fp16})", file=sys.stderr)


if __name__ == "__main__":
    main(*sys.argv[1:])
