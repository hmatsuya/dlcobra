"""Extract model weights from a Lightning checkpoint for use with resume_model.

Usage:
    python dlshogi/experiments/exp024_deep512_hao_data/extract_weights.py \
        dlshogi/wandb/wcsc36/s3lpnx8n/checkpoints/epoch=0-step=71250.ckpt \
        dlshogi/experiments/exp024_deep512_hao_data/exp023_best.pt
"""
import argparse
import torch
from collections import OrderedDict

def extract(ckpt_path, output_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]

    # Strip Lightning/compile prefixes: model._orig_mod.xxx -> xxx
    model_state = OrderedDict()
    for k, v in state_dict.items():
        # Remove "model._orig_mod." or "model." prefix
        if k.startswith("model._orig_mod."):
            new_key = k[len("model._orig_mod."):]
        elif k.startswith("model."):
            new_key = k[len("model."):]
        else:
            continue
        model_state[new_key] = v

    torch.save({"model": model_state}, output_path)
    print(f"Saved {len(model_state)} parameters to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", help="Lightning checkpoint path")
    parser.add_argument("output_path", help="Output .pt path")
    args = parser.parse_args()
    extract(args.ckpt_path, args.output_path)
