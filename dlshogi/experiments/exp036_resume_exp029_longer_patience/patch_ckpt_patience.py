#!/usr/bin/env python3
"""Patch a Lightning checkpoint's EarlyStopping state so training can resume.

Background
----------
exp029's ``last.ckpt`` was saved in a *stopped* state:
    wait_count=20, patience=20, stopping_reason=PATIENCE_EXHAUSTED (3)

Lightning's ``EarlyStopping.load_state_dict`` restores ``patience`` and
``wait_count`` from the checkpoint, so resuming with ``--ckpt_path`` ignores the
``patience: 50`` set in config.yaml and stops almost immediately (after one more
non-improving validation: 20 -> 21 >= 20).

This script copies the checkpoint and rewrites only the EarlyStopping callback
state, leaving model weights, optimizer moments, LR scheduler, and the global
step counter untouched. Resuming from the patched checkpoint therefore continues
the cosine LR decay from step=141250 while honouring the longer patience.

Usage
-----
    python patch_ckpt_patience.py <src.ckpt> <dst.ckpt> [--patience 50]
"""

import argparse

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("src", help="source checkpoint (read-only)")
    parser.add_argument("dst", help="destination checkpoint (patched copy)")
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="new EarlyStopping patience (default: 50)",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.src}")
    ckpt = torch.load(args.src, map_location="cpu", weights_only=False)

    callbacks = ckpt.get("callbacks")
    if not callbacks:
        raise SystemExit("No 'callbacks' state found in checkpoint; nothing to patch.")

    # EarlyStopping's state_key is dynamic, e.g. "EarlyStopping{'monitor': ...}".
    es_keys = [k for k in callbacks if k.startswith("EarlyStopping")]
    if not es_keys:
        raise SystemExit(
            "No EarlyStopping callback state found. Available callbacks:\n  "
            + "\n  ".join(callbacks.keys())
        )

    for key in es_keys:
        state = callbacks[key]
        before = {
            "wait_count": state.get("wait_count"),
            "patience": state.get("patience"),
            "stopping_reason": state.get("stopping_reason"),
        }
        # 0 == EarlyStoppingReason.NOT_STOPPED
        state["wait_count"] = 0
        state["patience"] = args.patience
        state["stopping_reason"] = 0
        state["stopping_reason_message"] = None
        print(f"Patched callback '{key}':")
        print(f"  before: {before}")
        print(
            f"  after : {{'wait_count': 0, 'patience': {args.patience}, "
            f"'stopping_reason': 0}}"
        )

    print(f"global_step (unchanged): {ckpt.get('global_step')}")
    print(f"epoch (unchanged):       {ckpt.get('epoch')}")

    print(f"Saving patched checkpoint: {args.dst}")
    torch.save(ckpt, args.dst)
    print("Done.")


if __name__ == "__main__":
    main()
