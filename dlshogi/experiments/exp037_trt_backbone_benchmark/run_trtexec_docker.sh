#!/usr/bin/env bash
# TensorRT FP16 latency benchmark via Docker (works around host driver/runtime mismatch).
#
# Uses NGC TensorRT container (24.05-py3: TRT 10.0 + CUDA 12.4) which is compatible
# with host driver 550.x. The container bundles its own trtexec and CUDA runtime.
#
# Workflow:
#   1. Export ONNX models (on host, using venv PyTorch)
#   2. Run trtexec inside container on exported ONNX files (GPU benchmark)
#   3. Parse results on host
#
# Usage:
#   bash run_trtexec_docker.sh [batch_sizes...]   (default: 16 64 256)
#
# Prerequisites:
#   - Docker with NVIDIA Container Toolkit (nvidia-ctk / --gpus)
#   - First run pulls ~13GB NGC image (cached thereafter)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

source "$REPO_ROOT/.venv/bin/activate"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

ONNXDIR="$SCRIPT_DIR/onnx"
RESDIR="$SCRIPT_DIR/results_trt"
mkdir -p "$ONNXDIR" "$RESDIR"

# TensorRT container: 24.05-py3 = TRT 10.0, CUDA 12.4, compatible with driver ≥550.54
TRT_IMAGE="nvcr.io/nvidia/tensorrt:24.05-py3"

NETWORKS=("inceptionnext" "resnet32x384" "senet32x384" "hybrid36p5" "hybrid37p4" "hybrid_exp038")
BATCHES=("$@")
if [ ${#BATCHES[@]} -eq 0 ]; then
    BATCHES=(16 64 256)
fi

echo "=== Host GPU ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "=== Container image: $TRT_IMAGE ==="
echo

# --- Step 1: Export ONNX models (on host) ---
echo "=== Exporting ONNX models ==="
for bs in "${BATCHES[@]}"; do
    for net in "${NETWORKS[@]}"; do
        onnx="$ONNXDIR/${net}_b${bs}.onnx"
        if [ -f "$onnx" ]; then
            echo "  [skip] $onnx exists"
            continue
        fi
        echo "  [export] $net batch=$bs"
        python "$SCRIPT_DIR/export_onnx.py" "$net" "$onnx" --batch "$bs" --fp16 \
            2> "$RESDIR/${net}_b${bs}.export.log" || \
            { echo "  [FAIL] export $net b$bs"; continue; }
    done
done

# --- Step 2: Run trtexec inside container ---
echo
echo "=== Running trtexec in Docker container ==="
echo "(first run will pull ~13GB image)"
echo

for bs in "${BATCHES[@]}"; do
    for net in "${NETWORKS[@]}"; do
        onnx="$ONNXDIR/${net}_b${bs}.onnx"
        log="$RESDIR/${net}_b${bs}.log"
        if [ ! -f "$onnx" ]; then
            continue
        fi
        echo ">>> trtexec $net batch=$bs (fp16, container)"
        docker run --rm --gpus all \
            -v "$ONNXDIR:/onnx:ro" \
            "$TRT_IMAGE" \
            trtexec --onnx="/onnx/${net}_b${bs}.onnx" --fp16 \
                --noDataTransfers --useSpinWait --useCudaGraph \
                --avgRuns=200 --warmUp=500 --duration=10 \
            > "$log" 2>&1 || { echo "  [FAIL] trtexec $net b$bs, see $log"; continue; }
        # Quick peek at result
        grep -i "GPU Compute Time:" "$log" | tail -1
    done
done

# --- Step 3: Parse results ---
echo
echo "=== Results ==="
python "$SCRIPT_DIR/parse_results.py" "$RESDIR"
