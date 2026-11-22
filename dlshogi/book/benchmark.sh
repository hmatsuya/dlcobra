#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON=${PYTHON:-"$ROOT/.venv/bin/python"}
ENGINE=${ENGINE:-"/home/hmatsuya/workspace/Shogi/YaneuraOuOriginal/exe/YaneuraOu-by-gcc"}
BOOK_DIR=${BOOK_DIR:-}
PUCT_BOOK=${PUCT_BOOK:-"$ROOT/output/punct_book.yaneuraou.db"}
TERASHOCK_BOOK=${TERASHOCK_BOOK:-"/home/hmatsuya/workspace/Shogi/data/terashock/user_book1.db"}
LIBRARY_PATH=${LIBRARY_PATH:-"/opt/onnxruntime-linux-x64-gpu-1.14.1/lib"}
GAMES=${GAMES:-30}
NODES=${NODES:-100000}
MAX_PLY=${MAX_PLY:-320}
TIMEOUT=${TIMEOUT:-120}
OUTPUT=${OUTPUT:-"$ROOT/output/book-benchmark.json"}
BOOK_ARGS=()
if [[ -n "$BOOK_DIR" ]]; then
  BOOK_ARGS=(--book-dir "$BOOK_DIR")
fi

exec \
  "$PYTHON" -m dlshogi.utils.book_benchmark \
  --engine "$ENGINE" \
  --puct-book "$PUCT_BOOK" \
  --terashock-book "$TERASHOCK_BOOK" \
  "${BOOK_ARGS[@]}" \
  --games "$GAMES" \
  --nodes "$NODES" \
  --concurrency "${CONCURRENCY:-8}" \
  --max-ply "$MAX_PLY" \
  --timeout "$TIMEOUT" \
  --output "$OUTPUT" \
  "$@"
