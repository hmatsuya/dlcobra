#!/usr/bin/env bash
# TensorRT FP16 latency benchmark: InceptionNeXt vs equal-param ResNet / SE-ResNet.
#
# Production inference path is ONNX -> TensorRT, so we export each net to ONNX
# (random weights; we measure speed only) and time it with trtexec in FP16.
#
# Networks (all ~86M params, fair comparison):
#   inceptionnext   86.14M  (exp026 architecture: depthwise 3x3 + 1x9 + 9x1)
#   resnet32x384    85.83M  (dense 3x3 convs)
#   senet32x384     87.02M  (dense 3x3 convs + SE channel attention)
#
# Usage: bash run_benchmark.sh [batch_sizes...]   (default: 1 16 64 256)
set -euo pipefail

cd "$(dirname "$0")/../../.."   # repo root
source .venv/bin/activate
export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

OUTDIR="dlshogi/experiments/exp037_trt_backbone_benchmark"
ONNXDIR="$OUTDIR/onnx"
RESDIR="$OUTDIR/results"
mkdir -p "$ONNXDIR" "$RESDIR"

NETWORKS=("inceptionnext" "resnet32x384" "senet32x384")
BATCHES=("$@")
if [ ${#BATCHES[@]} -eq 0 ]; then
    BATCHES=(1 16 64 256)
fi

echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "=== TensorRT ==="
trtexec --version 2>&1 | head -1
echo

for bs in "${BATCHES[@]}"; do
    for net in "${NETWORKS[@]}"; do
        onnx="$ONNXDIR/${net}_b${bs}.onnx"
        log="$RESDIR/${net}_b${bs}.log"
        echo ">>> export $net batch=$bs"
        python "$OUTDIR/export_onnx.py" "$net" "$onnx" --batch "$bs" --fp16 2> "$RESDIR/${net}_b${bs}.export.log"
        echo ">>> trtexec $net batch=$bs (fp16)"
        # This trtexec build is strongly-typed by default: precision is taken from
        # the ONNX graph (we export FP16). CUDA graph / spin-wait are on by default.
        trtexec --onnx="$onnx" \
            --avgRuns=200 --warmUp=500 --duration=10 \
            > "$log" 2>&1 || { echo "trtexec FAILED for $net b$bs, see $log"; continue; }
    done
done

echo
echo "=== Parsing results ==="
python "$OUTDIR/parse_results.py" "$RESDIR"
