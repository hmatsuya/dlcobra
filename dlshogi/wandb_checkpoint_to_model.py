from pytorch_lightning import LightningModule
import torch
from dlshogi import serializers

from ptl import Model
import wandb
import os
import yaml
import sys
import argparse

def main(*argv):

    artifact_path = "hmatsuya/wcsc25/model-o39a8ska:v29" # 1st lr cycle, 128ch 15 blocks

    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', default=artifact_path, help='artifact path')
    parser.add_argument('--model', default='model.npz', help='model file name')
    parser.add_argument('--fixed_batchsize', type=int, help='fixed batchsize')
    parser.add_argument('--downpath', default="download", help='download path')
    args = parser.parse_args(argv)

    # download wandb model checkpoint
    api = wandb.Api()
    artifact = api.artifact(args.artifact, type='model')
    downpath = artifact.download(args.downpath)

    # find file with .ckpt extension in downpath
    ckpt_file = [f for f in os.listdir(downpath) if f.endswith('.ckpt')][0]
    ckpt_path = os.path.join(downpath, ckpt_file)

    # Inspect the checkoint keys
    ckpt = torch.load(ckpt_path, map_location='cpu')
    # print("Checkpoint keys:", ckpt.keys())  # Usually contains 'state_dict', 'hyper_parameters', etc.
    # state_dict = ckpt['state_dict']
    # print("State dict keys (first 10):", list(state_dict.keys())[:10])

    # Load config.yaml
    config = None
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    # print("Config keys:", config.keys())
    # print("Config model:", config['model']['network'])

    # Compare with model's state dict
    model = Model(config['model']['network'])  # Initialize without loading checkpoint
    # print("Model's state dict keys (first 10):", list(model.state_dict().keys())[:10])

    ckpt_keys = set(ckpt['state_dict'].keys())  # Load checkpoint keys
    model_keys = set(model.state_dict().keys()) # Load model keys

    print("Checkpoint keys count:", len(ckpt_keys))
    print("Model keys count:", len(model_keys))

    # Find missing keys (present in checkpoint but not in model)
    missing_keys = ckpt_keys - model_keys
    print("\n\nMissing keys (checkpoint has them, but model doesn't):")
    print(sorted(missing_keys))

    # Find unexpected keys (present in model but not in checkpoint)
    unexpected_keys = model_keys - ckpt_keys
    print("\n\nUnexpected keys (model has them, but checkpoint doesn't):")
    print(sorted(unexpected_keys))

    # Find common keys (present in both)
    common_keys = ckpt_keys & model_keys
    print("\n\nCommon keys (present in both):")
    print(sorted(common_keys))

    try:
        model.load_state_dict(ckpt['state_dict'], strict=True)  # strict=True (default) raises error
    except RuntimeError as e:
        print("Error:", e)  # Prints missing/unexpected keys

    # load model from checkpoint
    # network_class_path = config['model']['network']
    model = Model.load_from_checkpoint(ckpt_path, config=config)

    serializers.save_npz(
        os.path.join(downpath, args.model),
        model.model,
    )

if __name__ == '__main__':
    main(*sys.argv[1:])