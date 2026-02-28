#!/bin/bash
source "$(dirname "$0")/../.venv/bin/activate"
python wandb_checkpoint_to_model.py
python convert_model_to_onnx.py download/model.npz model.onnx --network dlshogi.network.policy_value_network_cr.PolicyValueNetwork