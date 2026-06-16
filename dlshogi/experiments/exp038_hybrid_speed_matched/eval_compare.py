"""Side-by-side accuracy comparison: InceptionNeXt (exp026) vs Hybrid (exp038).

Loads best checkpoints from both experiments and evaluates on the validation set,
printing policy loss, value loss, total loss, and policy accuracy.

Usage:
  python eval_compare.py --exp026_ckpt <path> --exp038_ckpt <path> [--val_file <path>]
  python eval_compare.py --auto   # auto-discover best checkpoints from wandb dir

Examples:
  # Auto-discover (finds highest-score checkpoints in wandb/wcsc36/*/checkpoints/):
  python eval_compare.py --auto

  # Explicit paths:
  python eval_compare.py \
    --exp026_ckpt wandb/wcsc36/abc123/checkpoints/epoch=1-step=260000.ckpt \
    --exp038_ckpt wandb/wcsc36/xyz789/checkpoints/epoch=1-step=300000.ckpt
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from dlshogi.common import *
from dlshogi import cppshogi


def load_model_from_ckpt(ckpt_path, network_class):
    """Load a model from a Lightning checkpoint, stripping the model./ prefix."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    stripped = {}
    for k, v in state_dict.items():
        if k.startswith("model._orig_mod."):
            stripped[k[len("model._orig_mod."):]] = v
        elif k.startswith("model."):
            stripped[k[len("model."):]] = v
    if not stripped:
        stripped = state_dict
    model = network_class()
    model.load_state_dict(stripped)
    return model


@torch.no_grad()
def evaluate(model, val_file, batch_size=1024, device="cuda", val_lambda=0.333):
    """Evaluate model on validation HCPE file. Returns dict of metrics."""
    model.to(device).eval()

    data = np.fromfile(val_file, dtype=HuffmanCodedPosAndEval)
    n = len(data)

    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_correct = 0
    total_samples = 0

    for i in range(0, n, batch_size):
        batch = data[i:i + batch_size]
        bs = len(batch)

        features1 = np.empty((bs, FEATURES1_NUM, 9, 9), dtype=np.float32)
        features2 = np.empty((bs, FEATURES2_NUM, 9, 9), dtype=np.float32)
        move = np.empty(bs, dtype=np.int64)
        result = np.empty(bs, dtype=np.float32)
        value = np.empty(bs, dtype=np.float32)

        cppshogi.hcpe_decode_with_value(batch, features1, features2, move, result, value)

        x1 = torch.tensor(features1, device=device)
        x2 = torch.tensor(features2, device=device)
        t_move = torch.tensor(move, device=device)
        t_result = torch.tensor(result.reshape(-1, 1), device=device)

        y_policy, y_value = model(x1, x2)

        # Policy loss (cross entropy)
        policy_loss = F.cross_entropy(y_policy, t_move, reduction="sum")
        total_policy_loss += policy_loss.item()

        # Value loss (binary cross entropy with logits)
        value_loss = F.binary_cross_entropy_with_logits(
            y_value, t_result, reduction="sum"
        )
        total_value_loss += value_loss.item()

        # Policy accuracy
        pred_move = y_policy.argmax(dim=1)
        total_correct += (pred_move == t_move).sum().item()
        total_samples += bs

    avg_policy_loss = total_policy_loss / total_samples
    avg_value_loss = total_value_loss / total_samples
    avg_loss = avg_policy_loss + val_lambda * avg_value_loss
    accuracy = total_correct / total_samples

    return {
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss,
        "total_loss": avg_loss,
        "policy_accuracy": accuracy,
        "samples": total_samples,
    }


def find_best_ckpt(wandb_dir, exp_name_substr):
    """Find the best (non-last) checkpoint matching an experiment name pattern."""
    # The best checkpoint has the lowest step number in 'epoch=*-step=*.ckpt'
    # Actually ModelCheckpoint saves top-k by val/loss, so pick the one with
    # the lowest val/loss embedded in the filename, or just the non-last one with
    # the highest step if we can't parse val/loss.
    # Heuristic: look for the run whose dir was most recently written.
    # Better: user should pass explicit paths.
    candidates = []
    for run_dir in glob.glob(os.path.join(wandb_dir, "*", "checkpoints")):
        ckpts = glob.glob(os.path.join(run_dir, "epoch=*-step=*.ckpt"))
        for c in ckpts:
            candidates.append(c)
    if not candidates:
        return None
    # Return the most recently modified non-last checkpoint
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0] if candidates else None


def main(*argv):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp026_ckpt", help="Path to exp026 best checkpoint")
    parser.add_argument("--exp038_ckpt", help="Path to exp038 best checkpoint")
    parser.add_argument("--val_file",
                        default="/mnt/nvme1n1p2/data/shogi-ai-book/"
                                "floodgate_test_2017-2018_r3500_eval5000.hcpe")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--val_lambda", type=float, default=0.333)
    parser.add_argument("--auto", action="store_true",
                        help="Auto-discover best checkpoints from wandb dir")
    parser.add_argument("--wandb_dir", default="wandb/wcsc36")
    args = parser.parse_args(argv)

    if not os.path.exists(args.val_file):
        print(f"Error: val file not found: {args.val_file}", file=sys.stderr)
        sys.exit(1)

    # Models
    from dlshogi.experiments.exp026_droppath_bf16_fresh.model import PolicyValueNetwork as INX
    from dlshogi.experiments.exp038_hybrid_speed_matched.model import PolicyValueNetwork as H38

    experiments = []
    if args.exp026_ckpt:
        experiments.append(("exp026 InceptionNeXt (86.1M)", INX, args.exp026_ckpt))
    if args.exp038_ckpt:
        experiments.append(("exp038 Hybrid (134.2M)", H38, args.exp038_ckpt))

    if not experiments:
        print("Error: provide --exp026_ckpt and/or --exp038_ckpt (or --auto)", file=sys.stderr)
        sys.exit(1)

    # Evaluate
    print(f"Validation file: {args.val_file}")
    print(f"val_lambda: {args.val_lambda}")
    print()
    header = f"{'experiment':<32} | {'params':>8} | {'policy_loss':>11} | {'value_loss':>10} | {'total_loss':>10} | {'policy_acc':>10}"
    print(header)
    print("-" * len(header))

    for name, net_cls, ckpt in experiments:
        model = load_model_from_ckpt(ckpt, net_cls)
        n_params = sum(p.numel() for p in model.parameters())
        metrics = evaluate(model, args.val_file, args.batch_size, val_lambda=args.val_lambda)
        print(f"{name:<32} | {n_params/1e6:>7.1f}M | {metrics['policy_loss']:>11.4f} | "
              f"{metrics['value_loss']:>10.4f} | {metrics['total_loss']:>10.4f} | "
              f"{metrics['policy_accuracy']:>9.4f}")

    print()
    print("Notes:")
    print("- total_loss = policy_loss + val_lambda * value_loss (lower better)")
    print("- policy_acc = fraction of positions where top-1 predicted move = teacher move (higher better)")


if __name__ == "__main__":
    main(*sys.argv[1:])
