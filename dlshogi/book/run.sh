#!/usr/bin/env bash
# PUCT book pipeline: bounded search -> graceful stop -> propagate -> export.
#
# Usage: bash dlshogi/book/run.sh SEARCH_SECONDS [GPU_COUNT]
#   SEARCH_SECONDS  how long to search before the graceful stop (a positive integer)
#   GPU_COUNT       search processes, one per GPU starting at GPU 0 (default 2)
#
# Every GPU it will use must be free. A GPU held by another model makes
# onnxruntime fall back to the CPU without an error, which looks like a run
# but is far too slow to matter, so the script refuses to start instead.
# The previous export is copied to output/old/ before it is replaced.
set -uo pipefail
cd /home/hmatsuya/workspace/Shogi/dlcobra
ONNX="${ONNX:-/home/hmatsuya/workspace/Shogi/dlcobra/exp036_model.onnx}"
PY="${PY:-/home/hmatsuya/workspace/Shogi/dlcobra/.venv/bin/python}"
CFG=/tmp/config.yaml
cp dlshogi/book/config.yaml "$CFG"
TS_BOOK=/home/hmatsuya/workspace/Shogi/data/terashock/user_book1.db
SEARCH_SECONDS="${1:-}"
GPU_COUNT="${2:-2}"
die() { code="$1"; shift; echo "run.sh: $*" >&2; exit "$code"; }
is_positive_int() { case "$1" in ''|*[!0-9]*) return 1 ;; esac; [ "$((10#$1))" -ge 1 ]; }
is_positive_int "$SEARCH_SECONDS" || { echo "usage: bash $0 SEARCH_SECONDS [GPU_COUNT]  (SEARCH_SECONDS must be a positive integer)" >&2; exit 2; }
is_positive_int "$GPU_COUNT" || { echo "GPU_COUNT must be a positive integer, got '$GPU_COUNT'" >&2; exit 2; }
GPU_COUNT=$((10#$GPU_COUNT))   # '010' is ten, not octal eight
SEARCH_SECONDS=$((10#$SEARCH_SECONDS))
MIN_FREE_MIB="${MIN_FREE_MIB:-4000}"
case "$MIN_FREE_MIB" in ''|*[!0-9]*) echo "MIN_FREE_MIB must be a non-negative integer, got '$MIN_FREE_MIB'" >&2; exit 2 ;; esac
SIGINT_GRACE_STEPS="${SIGINT_GRACE_STEPS:-200}"   # 5 s each
is_positive_int "$SIGINT_GRACE_STEPS" || { echo "SIGINT_GRACE_STEPS must be a positive integer, got '$SIGINT_GRACE_STEPS'" >&2; exit 2; }
for gpu in $(seq 0 $((GPU_COUNT - 1))); do
    free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    case "$free" in
        ''|*[!0-9]*) echo "GPU $gpu: no usable reading from nvidia-smi (is there a GPU $gpu?); lower GPU_COUNT." >&2; exit 1 ;;
    esac
    if [ "$free" -lt "$MIN_FREE_MIB" ]; then
        echo "GPU $gpu has $free MiB free (need $MIN_FREE_MIB): unload whatever holds it, or lower GPU_COUNT." >&2
        exit 1
    fi
done
[ "${PREFLIGHT_ONLY:-0}" = 1 ] && { echo "preflight ok: $GPU_COUNT GPU(s) with >= $MIN_FREE_MIB MiB free"; exit 0; }
LOG=/tmp/punct_book_run.log
OUT="${OUT:-/home/hmatsuya/workspace/Shogi/dlcobra/output}"
mkdir -p "$OUT"
# Fetch the 700T shock book (YaneuraOu BOOK-700T-Shock, WCSC29) if not present.
if [ ! -f "$TS_BOOK" ]; then
    mkdir -p "$(dirname "$TS_BOOK")"
    curl -fsSL "https://github.com/yaneurao/YaneuraOu/releases/download/BOOK-700T-Shock/700T-shock-book.zip" \
        -o /tmp/100T-shock-book.zip \
        && "$PY" -m zipfile -e /tmp/100T-shock-book.zip /tmp/100T \
        && cp /tmp/100T/user_book1.db "$TS_BOOK" \
        || die 1 "could not fetch the Terashock book to $TS_BOOK"
fi
# Terashock import happens automatically inside `search` (once per book;
# source identity is tracked in book_meta). Explicit `import-terashock`
# would force a re-import every run.
export PYTHONUNBUFFERED=1
echo "=== search start $(date +%H:%M:%S) seconds=$SEARCH_SECONDS gpu_count=$GPU_COUNT ==="
"$PY" -m dlshogi.book search --config "$CFG" --gpu-count "$GPU_COUNT" --model "$ONNX" --log-path "$LOG" &
SEARCH_PID=$!
echo "search pid=$SEARCH_PID"
# SIGINT the search and give it SIGINT_GRACE_STEPS * 5 s to flush and exit.
# Returns 0 if it stopped by itself (its status is then left for `wait`), 1 if it had to be killed.
stop_search() {
    kill -INT "$SEARCH_PID" 2>/dev/null
    for _ in $(seq 1 "$SIGINT_GRACE_STEPS"); do kill -0 "$SEARCH_PID" 2>/dev/null || return 0; sleep 5; done
    kill -0 "$SEARCH_PID" 2>/dev/null || return 0   # it may have exited during the last sleep
    kill -9 "$SEARCH_PID" 2>/dev/null
    wait "$SEARCH_PID" 2>/dev/null
    return 1
}
# Ctrl-C, TERM or HUP aimed at this script: stop the search gracefully and quit
# without touching the previous export (the search would otherwise be orphaned).
on_interrupt() {
    trap '' INT TERM HUP
    echo "run.sh: interrupted; stopping the search, leaving the previous export untouched" >&2
    stop_search
    exit 130
}
trap on_interrupt INT TERM HUP
# Search has no budget of its own, so it exiting before the stop request means it failed.
elapsed=0
while [ "$elapsed" -lt "$SEARCH_SECONDS" ]; do
    kill -0 "$SEARCH_PID" 2>/dev/null || break
    sleep 1
    elapsed=$((elapsed + 1))
done
if ! kill -0 "$SEARCH_PID" 2>/dev/null; then
    wait "$SEARCH_PID"; rc=$?
    die 1 "search exited by itself after ${elapsed}s (status $rc); leaving the previous export untouched. See $LOG"
fi
echo "=== SIGINT at $(date +%H:%M:%S) ==="
stop_search || die 1 "search did not stop within the grace period and was killed; its last flush did not run, so at most the final flush interval of visit counts is lost (expansions and earlier flushes are stored). Previous export left untouched; run propagate and export by hand if you want to continue."
wait "$SEARCH_PID"; rc=$?
[ "$rc" -eq 0 ] || die 1 "search stopped with status $rc; leaving the previous export untouched. See $LOG"
echo "=== search stopped $(date +%H:%M:%S); propagate ==="
"$PY" -m dlshogi.book propagate --config "$CFG" \
    || die 1 "propagate failed (status $?); leaving the previous export untouched"
# Export to .new names and only touch the live pair once both came out non-empty:
# export exits 0 with zero records when nothing was propagated (for example a run
# too short for any edge to reach Propagation_Visit_Threshold), and would otherwise
# replace the good book with an empty file.
FILES="punct_book.apery punct_book.yaneuraou.db"
rm -f "$OUT"/punct_book.apery.new "$OUT"/punct_book.yaneuraou.db.new   # leftovers of an earlier failed run
echo "=== export $(date +%H:%M:%S) ==="
"$PY" -m dlshogi.book export --config "$CFG" \
    --apery "$OUT/punct_book.apery.new" --yaneuraou "$OUT/punct_book.yaneuraou.db.new" \
    || die 1 "export failed (status $?); the previous export is untouched"
for f in $FILES; do
    [ -s "$OUT/$f.new" ] || die 1 "export wrote an empty $f (nothing reached Propagation_Visit_Threshold? search longer); previous export untouched, new files left as $OUT/*.new"
done
mkdir -p "$OUT/old"
STAMP=$(date +%Y%m%d-%H%M%S)
for f in $FILES; do
    if [ -f "$OUT/$f" ]; then
        cp -p "$OUT/$f" "$OUT/old/$f.$STAMP" || die 1 "could not back up $OUT/$f; refusing to replace it"
    fi
done
for f in $FILES; do
    mv -f "$OUT/$f.new" "$OUT/$f" || die 1 "could not move $OUT/$f.new into place; the previous pair is in $OUT/old/*.$STAMP"
done
echo "=== PIPELINE DONE $(date +%H:%M:%S) ==="
