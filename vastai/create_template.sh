#!/bin/bash
# Creates (or updates) the Vast.ai template for the championship.
#
# Prerequisites:
#   1. vastai CLI installed and API key set  (vastai set api-key <key>)
#   2. Docker image built and pushed to DockerHub:
#        docker build -t <DOCKER_USER>/dlshogi-usi:latest -f vastai/Dockerfile .
#        docker push <DOCKER_USER>/dlshogi-usi:latest
#
# Usage:
#   DOCKER_USER=yourname bash vastai/create_template.sh
#
# To update an existing template instead of creating a new one:
#   TEMPLATE_HASH=<hash> DOCKER_USER=yourname bash vastai/create_template.sh

set -e

DOCKER_USER="${DOCKER_USER:-hmatsuya}"
IMAGE="${DOCKER_USER}/dlshogi-usi:latest"

# Minimum disk space in GB (model ~1 GB + TRT serialized cache + OS overhead)
DISK_GB=20

# Require CUDA >= 12.6 to match the base image (TRT 24.09)
EXTRA_FILTERS='{"cuda_max_good":{"gte":12.6}}'

# On-start: just print a ready message (engine is already compiled in the image)
ONSTART='echo "=== dlshogi USI engine ready ===" && echo "Binary: /usr/local/bin/usi" && echo "Model:  /workspace/model/model.onnx" && echo "Run engine: /workspace/run_usi.sh"'

if [ -n "${TEMPLATE_HASH}" ]; then
    # ── Update existing template ───────────────────────────────────────────────
    echo "Updating template $TEMPLATE_HASH ..."
    vastai update template "$TEMPLATE_HASH" \
        --image "$IMAGE" \
        --ssh \
        --direct \
        --onstart-cmd "$ONSTART" \
        --disk_space "$DISK_GB" \
        --desc "dlshogi USI engine (TensorRT inference, pre-compiled)" \
        --name "dlshogi-usi-championship"
else
    # ── Create new template ────────────────────────────────────────────────────
    echo "Creating new template with image: $IMAGE"
    vastai create template \
        --name "dlshogi-usi-championship" \
        --image "$IMAGE" \
        --ssh \
        --direct \
        --onstart-cmd "$ONSTART" \
        --disk_space "$DISK_GB" \
        --desc "dlshogi USI engine (TensorRT inference, pre-compiled). Upload ONNX model to /workspace/model/ after connecting."
fi

echo ""
echo "Done. Find your template at: https://cloud.vast.ai/templates/"
echo ""
echo "To launch an instance (example — RTX 4090, 1 GPU):"
echo "  vastai search offers 'gpu_name=RTX_4090 num_gpus=1 cuda_max_good>=12.6 disk_space>=$DISK_GB inet_down>=200' -o dph"
echo "  vastai create instance <OFFER_ID> --template_hash <HASH> --disk $DISK_GB"
