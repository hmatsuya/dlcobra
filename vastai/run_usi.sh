#!/bin/bash
# Launch the pre-compiled USI engine.
# The ONNX model and opening books are baked into the image.
#
# Usage:
#   /workspace/run_usi.sh [/path/to/model.onnx]
#
# Optional env vars (shown with defaults):
#   DNN_MODEL       /workspace/model/model.onnx
#   UCT_THREADS     2
#   DNN_BATCH_SIZE  128
#   OWN_BOOK        true
#   BOOK_FILE       /workspace/model/mafu_book.bin
#
# Available books in /workspace/model/:
#   mafu_book.bin   Mafu opening theory ver11 (Apery format, 22 MB) [default]
#   cobra_book.bin  Cobra book converted from YaneuraOu format (2.5 MB, 138K positions)

MODEL="${1:-${DNN_MODEL:-/workspace/model/model.onnx}}"
BOOK="${BOOK_FILE:-/workspace/model/mafu_book.bin}"
THREADS="${UCT_THREADS:-2}"
BATCH="${DNN_BATCH_SIZE:-128}"

if [ ! -f "$MODEL" ]; then
    echo "ERROR: model file not found: $MODEL" >&2
    exit 1
fi

MODEL_DIR="$(dirname "$MODEL")"
MODEL_FILE="$(basename "$MODEL")"
BOOK_FILE_NAME="$(basename "$BOOK")"

# The USI engine resolves DNN_Model and Book_File relative to its working
# directory. cd into the model directory so relative filenames work correctly.
cd "$MODEL_DIR" || exit 1

# Prepend setoption commands to stdin, then pass through the rest of stdin
# (USI GUI commands) to the engine.
(
  echo "setoption name DNN_Model value $MODEL_FILE"
  echo "setoption name Book_File value $BOOK_FILE_NAME"
  echo "setoption name UCT_Threads value $THREADS"
  echo "setoption name DNN_Batch_Size value $BATCH"
  cat  # pass through all subsequent stdin (USI GUI traffic)
) | exec /usr/local/bin/usi
