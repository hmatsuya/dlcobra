#!/bin/bash
# Converts a Lightning checkpoint to ONNX locally (no GPU needed with --gpu -1)
# and injects the resulting model.onnx into an existing Docker image.
#
# This keeps the 1.2 GB checkpoint OUT of the Docker build context, making
# `docker build` fast (~50 MB context, fully cacheable).
#
# Usage (from repo root):
#   bash vastai/inject_model.sh \
#     dlshogi/experiments/exp032_structured_pruning/pruned_state_dict.pt \
#     hmatsuya/dlshogi-usi:latest
#
# Prerequisites:
#   - Python venv active (source .venv/bin/activate)
#   - The target image already built (docker build ... -f vastai/Dockerfile .)

set -e

CKPT_PATH="${1}"
IMAGE="${2:-hmatsuya/dlshogi-usi:latest}"
ONNX_TMP="/tmp/dlshogi_model_$$.onnx"

if [ -z "$CKPT_PATH" ]; then
    echo "Usage: bash vastai/inject_model.sh <checkpoint.ckpt> [image:tag]"
    exit 1
fi

if [ ! -f "$CKPT_PATH" ]; then
    echo "ERROR: checkpoint not found: $CKPT_PATH"
    exit 1
fi

echo "=== Step 1: Export checkpoint → ONNX (CPU) ==="
echo "  checkpoint : $CKPT_PATH"
echo "  output     : $ONNX_TMP"
PYTHONPATH=. python vastai/export_onnx.py "$CKPT_PATH" "$ONNX_TMP" --gpu -1
echo "  ONNX size  : $(du -sh "$ONNX_TMP" | cut -f1)"

echo ""
echo "=== Step 2: Inject model.onnx into image: $IMAGE ==="

# Create a temporary container from the image, copy the file in, commit
CONTAINER_ID=$(docker create "$IMAGE")
docker cp "$ONNX_TMP" "${CONTAINER_ID}:/workspace/model/model.onnx"
docker commit \
    --message "inject model.onnx from $(basename "$CKPT_PATH")" \
    "$CONTAINER_ID" "$IMAGE"
docker rm "$CONTAINER_ID"

rm -f "$ONNX_TMP"

echo ""
echo "=== Done ==="
echo "  Image $IMAGE now contains /workspace/model/model.onnx"
echo ""
echo "Next step — push to Docker Hub:"
echo "  docker push $IMAGE"
