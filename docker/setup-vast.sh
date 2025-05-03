#!/bin/bash

# cd /workspace
# git clone https://github.com/hmatsuya/dlcobra.git

cd /workspace/dlcobra
git checkout wcsc35

touch ~/.no_auto_tmux

# update packages
apt update && apt upgrade -y
apt install -y aptitude

# # Add CUDA key ring
# cd /tmp
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
# dpkg -i cuda-keyring_1.1-1_all.deb
# apt update

# # install TensorRT
# # Not working. Use aptitude instead to install TensorRT.
# apt install -y libnvinfer-bin:amd64=8.6.1.6-1+cuda12.0
# apt install -y libnvinfer-dev:amd64=8.6.1.6-1+cuda12.0
# apt install -y libnvinfer-dispatch-dev=amd64 8.6.1.6-1+cuda12.0
# apt install -y libnvinfer-dispatch8:amd64=8.6.1.6-1+cuda12.0
# apt install -y libnvinfer-headers-dev:amd64=8.6.1.6-1+cuda12.0
# apt install -y libnvinfer-headers-plugin-dev:amd64=8.6.1.6-1+cuda12.0
# apt install -y libnvinfer-lean-dev:amd64=8.6.1.6-1+cuda12.0
# apt install -y libnvinfer-lean8:amd64=8.6.1.6-1+cuda12.0
# apt install -y libnvinfer-plugin-dev:amd64=8.6.1.6-1+cuda12.0
# apt install -y libnvinfer-samples:amd64=8.6.1.6-1+cuda12.0
# apt install -y libnvinfer-vc-plugin-dev:amd64=8.6.1.6-1+cuda12.0
# apt install -y libnvinfer-vc-plugin8:amd64=8.6.1.6-1+cuda12.0
# apt install -y libnvonnxparsers-dev:amd64=8.6.1.6-1+cuda12.0
# apt install -y libnvonnxparsers8:amd64=8.6.1.6-1+cuda12.0
# apt install -y libnvparsers-dev:amd64=8.6.1.6-1+cuda12.0
# apt install -y libnvparsers8:amd64=8.6.1.6-1+cuda12.0
# apt install -y tensorrt:amd64=8.6.1.6-1+cuda12.0


# build the executable
cd /workspace/dlcobra/usi
make -j

# copy the model
cp /workspace/dlcobra/dlshogi/model.onnx /workspace/dlcobra/usi/bin/


cd /workspace/dlcobra/usi/bin


# ========================================
# [INSTALL, DEPENDENCIES] libnvinfer-bin:amd64 8.6.1.6-1+cuda12.0
# [INSTALL, DEPENDENCIES] libnvinfer-dev:amd64 8.6.1.6-1+cuda12.0
# [INSTALL, DEPENDENCIES] libnvinfer-dispatch-dev:amd64 8.6.1.6-1+cuda12.0
# [INSTALL, DEPENDENCIES] libnvinfer-dispatch8:amd64 8.6.1.6-1+cuda12.0
# [INSTALL, DEPENDENCIES] libnvinfer-headers-dev:amd64 8.6.1.6-1+cuda12.0
# [INSTALL, DEPENDENCIES] libnvinfer-headers-plugin-dev:amd64 8.6.1.6-1+cuda12.0
# [INSTALL, DEPENDENCIES] libnvinfer-lean-dev:amd64 8.6.1.6-1+cuda12.0
# [INSTALL, DEPENDENCIES] libnvinfer-lean8:amd64 8.6.1.6-1+cuda12.0
# [INSTALL, DEPENDENCIES] libnvinfer-plugin-dev:amd64 8.6.1.6-1+cuda12.0
# [INSTALL, DEPENDENCIES] libnvinfer-samples:amd64 8.6.1.6-1+cuda12.0
# [INSTALL, DEPENDENCIES] libnvinfer-vc-plugin-dev:amd64 8.6.1.6-1+cuda12.0
# [INSTALL, DEPENDENCIES] libnvinfer-vc-plugin8:amd64 8.6.1.6-1+cuda12.0
# [INSTALL, DEPENDENCIES] libnvonnxparsers-dev:amd64 8.6.1.6-1+cuda12.0
# [INSTALL, DEPENDENCIES] libnvonnxparsers8:amd64 8.6.1.6-1+cuda12.0
# [INSTALL, DEPENDENCIES] libnvparsers-dev:amd64 8.6.1.6-1+cuda12.0
# [INSTALL, DEPENDENCIES] libnvparsers8:amd64 8.6.1.6-1+cuda12.0
# [INSTALL] tensorrt:amd64 8.6.1.6-1+cuda12.0
# ========================================


