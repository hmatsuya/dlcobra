import dlshogi.convert_model_to_onnx

# convert to onnx model file
name = "dlshogi.network.policy_value_network_effnet10.PolicyValueNetwork"
model_path = "/home/hmatsuya/workspace/Shogi/cobra-shibori/model/model_dlshogi.network.policy_value_network_effnet10.PolicyValueNetwork-finetuning-2022-08-16T090547"

# model_path = "/home/hmatsuya/workspace/Shogi/cobra-shibori/model/model_dlshogi.network.policy_value_network_effnet10.PolicyValueNetwork-000"

dlshogi.convert_model_to_onnx.main(*['--network', name, model_path, model_path + '.onnx'])
