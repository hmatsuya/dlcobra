import onnx

model = onnx.load('bin/model.onnx')
for input in model.graph.input:
    print(input.name)