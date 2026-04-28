"""Structured pruning script for exp032.

Loads the exp029 best checkpoint and prunes the MLP expansion layers
(pwconv1 / pwconv2) in all 39 InceptionNeXtBlocks using L1-norm channel
importance scoring.  The attention block and all other layers are untouched.

Usage:
    python dlshogi/experiments/exp032_structured_pruning/prune.py \\
        --ckpt /path/to/epoch=0-step=116250.ckpt \\
        --output /path/to/pruned_state_dict.pt \\
        --prune-ratio 0.25

    # Quick sanity check (dry-run, no file written):
    python dlshogi/experiments/exp032_structured_pruning/prune.py \\
        --ckpt /path/to/epoch=0-step=116250.ckpt \\
        --prune-ratio 0.25 \\
        --dry-run

Prune ratio guide:
    0.10 → 10% removed, expansion 2048 → 1840  (77.8M params)
    0.25 → 25% removed, expansion 2048 → 1536  (65.7M params)  ← recommended start
    0.40 → 40% removed, expansion 2048 → 1224  (53.2M params)  ← aggressive
    0.50 → 50% removed, expansion 2048 → 1024  (45.2M params)  ← very aggressive
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Resolve repo root so this script works when run from any directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[3]  # .../DeepLearningShogi
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dlshogi.experiments.exp026_droppath_bf16_fresh.model import PolicyValueNetwork as OriginalNetwork
from dlshogi.experiments.exp032_structured_pruning.model import PolicyValueNetwork as PrunedNetwork


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_original(ckpt_path: str) -> OriginalNetwork:
    """Load exp026/exp029 checkpoint into the original (unpruned) network.

    exp029 checkpoints contain both the student model (model.*) and the
    teacher model (_teacher_model.*) in the same state dict.  We extract
    only the student weights and ignore everything else.
    """
    model = OriginalNetwork()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)

    # Extract only keys that belong to the student model, stripping prefixes:
    #   "model._orig_mod." → torch.compile wrapping
    #   "model."           → standard Lightning wrapping
    # Keys starting with "_teacher_model." are discarded.
    stripped = {}
    for k, v in state.items():
        if k.startswith("model._orig_mod."):
            stripped[k[len("model._orig_mod."):]] = v
        elif k.startswith("model."):
            stripped[k[len("model."):]] = v
        # else: _teacher_model.*, hparams, etc. — skip

    missing, unexpected = model.load_state_dict(stripped, strict=False)
    if missing:
        raise RuntimeError(f"Missing keys when loading checkpoint: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys when loading checkpoint: {unexpected}")
    model.eval()
    return model


def l1_scores(weight: torch.Tensor) -> torch.Tensor:
    """L1 norm of each output channel (dim 0) — used to rank importance."""
    return weight.abs().sum(dim=tuple(range(1, weight.ndim)))


def compute_keep_indices(pwconv1_weight: torch.Tensor, n_keep: int) -> torch.Tensor:
    """Return sorted indices of the n_keep most important output channels."""
    scores = l1_scores(pwconv1_weight)
    keep_idx = scores.argsort(descending=True)[:n_keep]
    keep_idx, _ = keep_idx.sort()
    return keep_idx


def prune_block_weights(
    src_block,
    dst_block,
    n_keep: int,
) -> None:
    """Copy pruned weights from src InceptionNeXtBlock into dst InceptionNeXtBlock.

    Pruned dimensions:
        pwconv1: (mlp_dim, dim) → (n_keep, dim)
        pwconv2: (dim, mlp_dim) → (dim, n_keep)
    All other sub-modules (depthwise convs, norm) are copied as-is.
    """
    # Depthwise convs and norm: straight copy
    dst_block.dw3x3.load_state_dict(src_block.dw3x3.state_dict())
    dst_block.dw1x9.load_state_dict(src_block.dw1x9.state_dict())
    dst_block.dw9x1.load_state_dict(src_block.dw9x1.state_dict())
    dst_block.norm.load_state_dict(src_block.norm.state_dict())
    if hasattr(src_block.drop_path, "drop_prob"):
        dst_block.drop_path.drop_prob = src_block.drop_path.drop_prob

    # Compute keep indices from pwconv1 output channels
    keep_idx = compute_keep_indices(src_block.pwconv1.weight, n_keep)

    with torch.no_grad():
        # pwconv1: keep selected output channels
        dst_block.pwconv1.weight.copy_(src_block.pwconv1.weight[keep_idx])
        if src_block.pwconv1.bias is not None:
            dst_block.pwconv1.bias.copy_(src_block.pwconv1.bias[keep_idx])

        # pwconv2: keep corresponding input channels
        dst_block.pwconv2.weight.copy_(src_block.pwconv2.weight[:, keep_idx])
        if src_block.pwconv2.bias is not None:
            dst_block.pwconv2.bias.copy_(src_block.pwconv2.bias)


def build_pruned_model(src: OriginalNetwork, prune_ratio: float) -> PrunedNetwork:
    """Build a pruned PrunedNetwork from a trained OriginalNetwork.

    Args:
        src: Trained original model (expansion_dim = 2048).
        prune_ratio: Fraction of MLP channels to remove (e.g. 0.25).

    Returns:
        PrunedNetwork with smaller pwconv1/pwconv2 and weights transferred.
    """
    dim = 512
    original_mlp_dim = dim * 4  # 2048
    n_keep = int(original_mlp_dim * (1.0 - prune_ratio))
    # Round down to nearest multiple of 8 for hardware alignment
    n_keep = (n_keep // 8) * 8
    print(f"MLP dim: {original_mlp_dim} → {n_keep}  (keeping {n_keep/original_mlp_dim*100:.1f}%)")

    dst = PrunedNetwork(mlp_expansion_dim=n_keep)
    dst.eval()

    # --- Stem layers: straight copy ---
    dst.stem_1.load_state_dict(src.stem_1.state_dict())
    dst.stem_2.load_state_dict(src.stem_2.state_dict())
    dst.stem_norm.load_state_dict(src.stem_norm.state_dict())

    # --- Blocks ---
    src_blocks = list(src.blocks.children())
    dst_blocks = list(dst.blocks.children())
    assert len(src_blocks) == len(dst_blocks), (
        f"Block count mismatch: src={len(src_blocks)}, dst={len(dst_blocks)}"
    )

    from dlshogi.experiments.exp026_droppath_bf16_fresh.model import InceptionNeXtBlock as SrcBlock
    from dlshogi.experiments.exp032_structured_pruning.model import InceptionNeXtBlock as DstBlock

    for i, (sb, db) in enumerate(zip(src_blocks, dst_blocks)):
        if isinstance(sb, SrcBlock):
            assert isinstance(db, DstBlock)
            prune_block_weights(sb, db, n_keep)
        else:
            # PlainSelfAttentionBlock — copy as-is
            db.load_state_dict(sb.state_dict())

    # --- Heads: straight copy ---
    dst.policy.load_state_dict(src.policy.state_dict())
    with torch.no_grad():
        dst.policy_bias.copy_(src.policy_bias)
    dst.value_conv.load_state_dict(src.value_conv.state_dict())
    dst.value_norm.load_state_dict(src.value_norm.state_dict())
    dst.value_fc1.load_state_dict(src.value_fc1.state_dict())
    dst.value_fc2.load_state_dict(src.value_fc2.state_dict())

    return dst, n_keep


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def quick_forward_check(model: nn.Module, device: str = "cpu") -> None:
    """Run a single forward pass to verify the pruned model is functional."""
    from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
    x1 = torch.zeros(2, FEATURES1_NUM, 9, 9, device=device)
    x2 = torch.zeros(2, FEATURES2_NUM, 9, 9, device=device)
    with torch.no_grad():
        y1, y2 = model(x1, x2)
    assert y1.shape == (2, 9 * 9 * 27), f"Unexpected policy shape: {y1.shape}"
    assert y2.shape == (2, 1), f"Unexpected value shape: {y2.shape}"
    print("Forward check passed ✓")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Structured pruning for exp032")
    parser.add_argument(
        "--ckpt",
        default="/home/hmatsuya/workspace/Shogi/dlcobra/dlshogi/wandb/wcsc36/wkpn9zzz/checkpoints/epoch=0-step=116250.ckpt",
        help="Path to exp029 best checkpoint (Lightning .ckpt format)",
    )
    parser.add_argument(
        "--output",
        default=str(_SCRIPT_DIR / "pruned_state_dict.pt"),
        help="Output path for pruned state dict (.pt)",
    )
    parser.add_argument(
        "--prune-ratio",
        type=float,
        default=0.25,
        help="Fraction of MLP channels to remove (default: 0.25)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pruning and checks but do not write output file",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    src = load_original(args.ckpt)
    before = count_params(src)
    print(f"Original parameters: {before / 1e6:.2f}M")

    print(f"\nPruning MLP channels (prune_ratio={args.prune_ratio:.0%}) ...")
    dst, n_keep = build_pruned_model(src, prune_ratio=args.prune_ratio)
    after = count_params(dst)
    reduction = (before - after) / before
    print(f"Pruned parameters:   {after / 1e6:.2f}M  ({reduction:.1%} reduction)")
    print(f"mlp_expansion_dim for config.yaml: {n_keep}")

    print("\nRunning forward check ...")
    quick_forward_check(dst)

    if not args.dry_run:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Wrap in {"model": ...} so ptl.py's resume_model loader recognises it
        # (same format as serializers.save_npz, handled by the "model" in checkpoint branch)
        torch.save({"model": dst.state_dict()}, output_path)
        print(f"\nSaved pruned state dict → {output_path}")
        print(
            f"\nNext step: update config.yaml with:\n"
            f"  model:\n"
            f"    network: dlshogi.experiments.exp032_structured_pruning.model.PolicyValueNetwork\n"
            f"    resume_model: {output_path}\n"
            f"    # mlp_expansion_dim is baked into the saved weights;\n"
            f"    # pass it via network init_args if using LightningCLI directly.\n"
        )
    else:
        print("\nDry run — no file written.")


if __name__ == "__main__":
    main()
