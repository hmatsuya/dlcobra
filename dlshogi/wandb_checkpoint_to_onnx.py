from pytorch_lightning import LightningModule
import torch
import torch.onnx
import onnx
from dlshogi.common import FEATURES1_NUM, FEATURES2_NUM, MAX_MOVE_LABEL_NUM
from dlshogi.network.policy_value_network import policy_value_network
from ptl import Model
import wandb
import os
import yaml
import sys
import argparse

def main(*argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', default='model.onnx', help='model file name')
    parser.add_argument('--fixed_batchsize', type=int)
    args = parser.parse_args(argv)

    # artifact_path = "hmatsuya/mofushogi/model-5c0vcfep:v24" # 256ch
    # artifact_path = "hmatsuya/mofushogi/model-gt0wfwia:v42" # 64ch
    # artifact_path = "hmatsuya/wcsc25/model-2t3qsi0f:v88" # 128ch 15 blocks
    artifact_path = "hmatsuya/wcsc25/model-9tnf6mth:v0" # test 128ch 15 blocks

    # download wandb model checkpoint
    api = wandb.Api()
    artifact = api.artifact(artifact_path, type='model')
    downpath = artifact.download('download')

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
    model = model.to('cpu')
    # model.fuse()
    model.eval()

    # export to onnx
    params = model.state_dict()
    dummy_input1 = torch.randn(1, FEATURES1_NUM, 9, 9, device='cpu')  # FEATURES1_NUM
    dummy_input2 = torch.randn(1, FEATURES2_NUM, 9, 9, device='cpu')   # FEATURES2_NUM (adjust if needed)
    print(f"FEATURES1_NUM: {FEATURES1_NUM}, FEATURES2_NUM: {FEATURES2_NUM}")

    if args.fixed_batchsize is None:
        torch.onnx.export(model, (dummy_input1, dummy_input2), args.onnx,
            params=params,
            dynamo=False,
            verbose = True,
            report=True,
            do_constant_folding = True,
            input_names = ['input1', 'input2'],
            output_names = ['output_policy', 'output_value'],
            dynamic_axes={
                'input1' : {0 : 'batch_size'},
                'input2' : {0 : 'batch_size'},
                'output_policy' : {0 : 'batch_size'},
                'output_value' : {0 : 'batch_size'},
                })
    else:
        torch.onnx.export(model, (dummy_input1, dummy_input2), args.onnx,
            params=params,
            dynamo=False,
            verbose = True,
            report=True,
            do_constant_folding = True,
            input_names = ['input1', 'input2'],
            output_names = ['output_policy', 'output_value'])

    # Load the ONNX model
    onnx_model = onnx.load("model.onnx")

    # Check if the model is well-formed
    onnx.checker.check_model(onnx_model)

    # Print the model's graph
    # print(onnx.helper.print_graph_text(onnx_model.graph))

if __name__ == '__main__':
    main(*sys.argv[1:])