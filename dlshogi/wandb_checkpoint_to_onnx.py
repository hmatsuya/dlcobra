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

POSITION_CHANNELS = 42
# artifact_path = "hmatsuya/mofushogi/model-5c0vcfep:v24" # 256ch
# artifact_path = "hmatsuya/mofushogi/model-gt0wfwia:v42" # 64ch
artifact_path = "hmatsuya/wcsc25/model-2t3qsi0f:v88"

# download wandb model checkpoint
api = wandb.Api()
artifact = api.artifact(artifact_path, type='model')
downpath = artifact_dir = artifact.download('download')

# find file with .ckpt extension in downpath
ckpt_file = [f for f in os.listdir(downpath) if f.endswith('.ckpt')][0]
ckpt_path = os.path.join(downpath, ckpt_file)

# Load config.yaml
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# load model from checkpoint
# network_class_path = config['model']['network']
model = Model.load_from_checkpoint(ckpt_path, config=config)
model = model.to('cpu')
model.eval()

# export to onnx
params = model.state_dict()
dummy_input1 = torch.randn(1, FEATURES1_NUM, 9, 9, device='cpu')  # FEATURES1_NUM
dummy_input2 = torch.randn(1, FEATURES2_NUM, 9, 9, device='cpu')   # FEATURES2_NUM (adjust if needed)
torch.onnx.export(
    model,
    (dummy_input1, dummy_input2),
    "model.onnx",
    params=params,
    dynamo=False,
    report=True,
    input_names=['features1', 'features2'],
    output_names=['policy_output', 'value_output']
)

# Load the ONNX model
onnx_model = onnx.load("model.onnx")

# Check if the model is well-formed
onnx.checker.check_model(onnx_model)

# Print the model's graph
# print(onnx.helper.print_graph_text(onnx_model.graph))