"""Export the exp032 pruned model to ONNX.

Usage (from repo root, with venv active):

  # Preferred — load the pruned state dict directly (fast, no prefix stripping):
  python vastai/export_onnx.py \
      dlshogi/experiments/exp032_structured_pruning/pruned_state_dict.pt \
      output.onnx [--gpu 0]

  # Alternative — load from a Lightning checkpoint (strips "model." prefix):
  python vastai/export_onnx.py \
      dlshogi/wandb/wcsc36/wkpn9zzz/checkpoints/last.ckpt \
      output.onnx [--gpu 0]

The script:
  1. Instantiates PrunedPolicyValueNetwork (mlp_expansion_dim=1536)
  2. Loads weights from a .pt state dict OR a Lightning .ckpt
  3. Exports to ONNX with dynamic batch axis and named I/O tensors that
     match what nn_tensorrt.cpp expects: input1, input2, output_policy, output_value
"""
import argparse
import sys

import torch
import torch.nn.functional as F

# Repo root must be on PYTHONPATH (handled by Dockerfile WORKDIR + PYTHONPATH)
from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM
from dlshogi.experiments.exp032_structured_pruning.pruned_network import PrunedPolicyValueNetwork


def load_checkpoint(ckpt_path: str, model: torch.nn.Module, device: torch.device) -> None:
    """Load weights from a pruned .pt state dict or a Lightning .ckpt file."""
    raw = torch.load(ckpt_path, map_location="cpu")

    # Lightning checkpoint has a "state_dict" key; plain .pt files do not
    if "state_dict" in raw:
        state_dict = raw["state_dict"]
        # Strip "model._orig_mod." or "model." prefixes added by Lightning/compile.
        # Also drop "_teacher_model.*" keys present in knowledge-distillation runs.
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith("_teacher_model."):
                continue  # frozen teacher — not needed for inference
            elif k.startswith("model._orig_mod."):
                stripped[k[len("model._orig_mod."):]] = v
            elif k.startswith("model."):
                stripped[k[len("model."):]] = v
            else:
                stripped[k] = v
        state_dict = stripped
    elif "model" in raw and hasattr(raw["model"], "keys"):
        # pruned_state_dict.pt saved as {"model": OrderedDict(...)}
        state_dict = raw["model"]
    else:
        # Plain state dict
        state_dict = raw

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()


def make_dummy_inputs(device: torch.device, batch_size: int = 1):
    x1 = torch.zeros(batch_size, FEATURES1_NUM, 9, 9, dtype=torch.float32, device=device)
    x2 = torch.zeros(batch_size, FEATURES2_NUM, 9, 9, dtype=torch.float32, device=device)
    return x1, x2


def export(ckpt_path: str, onnx_path: str, gpu: int) -> None:
    device = torch.device(f"cuda:{gpu}" if gpu >= 0 and torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = PrunedPolicyValueNetwork()
    print(f"Loading checkpoint: {ckpt_path}")
    load_checkpoint(ckpt_path, model, device)

    x1, x2 = make_dummy_inputs(device)

    print(f"Exporting to: {onnx_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            (x1, x2),
            onnx_path,
            # Opset 17: LayerNormalization is a first-class op, which TRT 10+
            # handles efficiently as a fused kernel.
            # NOTE: TRT 8.6 cannot parse LayerNormalization from opset 17 —
            # use opset 16 if targeting TRT 8.x.
            opset_version=17,
            do_constant_folding=True,
            input_names=["input1", "input2"],
            output_names=["output_policy", "output_value"],
            dynamic_axes={
                "input1":        {0: "batch_size"},
                "input2":        {0: "batch_size"},
                "output_policy": {0: "batch_size"},
                "output_value":  {0: "batch_size"},
            },
            dynamo=False,   # legacy exporter — produces IR v7/8
        )

    # torch.onnx.export may write weights to an external .data file when the
    # model is large. Inline them back into a single self-contained .onnx file
    # so the USI engine (and docker cp) only needs one file.
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model
    import os, shutil, tempfile

    data_file = onnx_path + ".data"
    if os.path.exists(data_file):
        print("Inlining external weights into single ONNX file...")
        m = onnx.load(onnx_path)
        load_external_data_for_model(m, os.path.dirname(onnx_path))
        # Save to a temp path then replace original
        tmp = onnx_path + ".tmp"
        onnx.save(m, tmp)
        os.replace(tmp, onnx_path)
        os.remove(data_file)
        print(f"  Final size: {os.path.getsize(onnx_path)/1e6:.1f} MB")

    print("ONNX export complete.")

    # Quick sanity check
    import onnx
    m = onnx.load(onnx_path)
    onnx.checker.check_model(m)
    print("ONNX model check passed.")


def main():
    parser = argparse.ArgumentParser(description="Export exp032 best ckpt to ONNX")
    parser.add_argument("checkpoint", help="Path to Lightning .ckpt file")
    parser.add_argument("onnx", help="Output .onnx path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID (-1 for CPU)")
    args = parser.parse_args()
    export(args.checkpoint, args.onnx, args.gpu)


if __name__ == "__main__":
    main()
