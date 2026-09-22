#!/usr/bin/env bash
# PUCT book pipeline on GPU 0 only (GPU 1 is running HyperQwen).
# bounded search -> graceful stop -> propagate -> export.
set -uo pipefail
cd /home/hmatsuya/workspace/Shogi/dlcobra
ONNX=/home/hmatsuya/workspace/Shogi/dlcobra/exp036_model.onnx
PY=/home/hmatsuya/workspace/Shogi/dlcobra/.venv/bin/python
CFG=/tmp/config.yaml
cp dlshogi/book/config.yaml "$CFG"
TS_BOOK=/home/hmatsuya/workspace/Shogi/data/terashock/user_book1.db
SEARCH_SECONDS="${1:-0}"
GPU_COUNT="${2:-2}"
LOG=/tmp/punct_book_run.log
OUT=/home/hmatsuya/workspace/Shogi/dlcobra/output
mkdir -p "$OUT"
# Fetch the 700T shock book (YaneuraOu BOOK-700T-Shock, WCSC29) if not present.
if [ ! -f "$TS_BOOK" ]; then
    mkdir -p "$(dirname "$TS_BOOK")"
    curl -fsSL "https://github.com/yaneurao/YaneuraOu/releases/download/BOOK-700T-Shock/700T-shock-book.zip" \
        -o /tmp/100T-shock-book.zip
    "$PY" -m zipfile -e /tmp/100T-shock-book.zip /tmp/100T
    cp /tmp/100T/user_book1.db "$TS_BOOK"
fi
# Terashock import happens automatically inside `search` (once per book;
# source identity is tracked in book_meta). Explicit `import-terashock`
# would force a re-import every run.
export PYTHONUNBUFFERED=1
echo "=== search start $(date +%H:%M:%S) seconds=$SEARCH_SECONDS gpu_count=$GPU_COUNT ==="
"$PY" -m dlshogi.book search --config "$CFG" --gpu-count "$GPU_COUNT" --model "$ONNX" --log-path "$LOG" &
SEARCH_PID=$!
echo "search pid=$SEARCH_PID"
sleep "$SEARCH_SECONDS"
echo "=== SIGINT at $(date +%H:%M:%S) ==="
kill -INT "$SEARCH_PID" 2>/dev/null || true
for i in $(seq 1 200); do kill -0 "$SEARCH_PID" 2>/dev/null || break; sleep 5; done
kill -9 "$SEARCH_PID" 2>/dev/null || true
echo "=== search stopped $(date +%H:%M:%S); propagate ==="
"$PY" -m dlshogi.book propagate --config "$CFG"
echo "=== export $(date +%H:%M:%S) ==="
"$PY" -m dlshogi.book export --config "$CFG" \
  --apery "$OUT/punct_book.apery" --yaneuraou "$OUT/punct_book.yaneuraou.db"
echo "=== PIPELINE DONE $(date +%H:%M:%S) ==="
