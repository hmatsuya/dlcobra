#!/bin/bash
pipenv run python wandb_checkpoint_to_model.py
pipenv run python convert_model_to_onnx.py download/model.npz model.onnx --network dlshogi.network.policy_value_network_cr.PolicyValueNetwork