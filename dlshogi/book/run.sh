#!/usr/bin/env bash
# PUCT book pipeline: bounded search -> graceful stop -> propagate -> export.
#
# Usage: bash dlshogi/book/run.sh [SEARCH_SECONDS] [GPU_COUNT]   (both optional)
#   SEARCH_SECONDS  how long to search before the graceful stop (a positive integer;
#                   default 2592000 = 30 days, so the normal way to end a run is Ctrl-C)
#   GPU_COUNT       search processes, one per GPU starting at GPU 0 (default 2)
#
# Every GPU it will use must be free. If model services hold the GPUs, the
# script stops those services before retrying, then restarts only services that
# were running when this script started.
# The previous export is copied to output/old/ before it is replaced.
set -uo pipefail
cd /home/hmatsuya/workspace/Shogi/dlcobra
ONNX="${ONNX:-/home/hmatsuya/workspace/Shogi/dlcobra/exp036_model.onnx}"
PY="${PY:-/home/hmatsuya/workspace/Shogi/dlcobra/.venv/bin/python}"
CFG=/tmp/config.yaml
TS_BOOK=/home/hmatsuya/workspace/Shogi/data/terashock/user_book1.db
BOOK_LOCK_DIR=/home/hmatsuya/workspace/Shogi/dlcobra/output/.puct-book
BOOK_LOCK_FILE="$BOOK_LOCK_DIR/book.lock"
SEARCH_SECONDS="${1:-2592000}"   # 30 days; Ctrl-C (or TERM/HUP) to stop early
GPU_COUNT="${2:-2}"
die() { code="$1"; shift; echo "run.sh: $*" >&2; exit "$code"; }

if [ -L "$BOOK_LOCK_DIR" ] || { [ -e "$BOOK_LOCK_DIR" ] && [ ! -d "$BOOK_LOCK_DIR" ]; }; then
    die 1 "book lock directory is not a directory: $BOOK_LOCK_DIR"
fi
mkdir -p -- "$BOOK_LOCK_DIR" || die 1 "could not create book lock directory: $BOOK_LOCK_DIR"
chmod 700 -- "$BOOK_LOCK_DIR" || die 1 "could not secure book lock directory: $BOOK_LOCK_DIR"
exec 9>"$BOOK_LOCK_FILE" || die 1 "could not open book lock: $BOOK_LOCK_FILE"
flock -n 9 || die 1 "another book build is already running"
CFG=/tmp/config.yaml
cp dlshogi/book/config.yaml "$CFG"

STRATA_PATTERN="/home/hmatsuya/opt/strata-orca/Strata/.venv/bin/python -m serve.server --engine strata"
STRATA_START=/home/hmatsuya/opt/strata-orca/run-orca.sh
LLAMA_SWAP=/home/hmatsuya/opt/llama-swap/llama-swap.sh
STRATA_WAS_RUNNING=0
LLAMA_WAS_RUNNING=0
SERVICES_STOPPED=0

check_service_scripts() {
    [ -x "$STRATA_START" ] || { echo "service script missing or not executable: $STRATA_START" >&2; return 1; }
    [ -x "$LLAMA_SWAP" ] || { echo "service script missing or not executable: $LLAMA_SWAP" >&2; return 1; }
}

# Direct preflight does not manage services, so it does not require their scripts.
[ "${PREFLIGHT_ONLY:-0}" = 1 ] || check_service_scripts || die 1 "cannot manage model services"

strata_pids() { pgrep -f -- "^$STRATA_PATTERN" || true; }
strata_running() { [ -n "$(strata_pids)" ]; }
llama_running() { docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'llama-swap'; }

stop_model_services() {
    check_service_scripts || return 1
    STRATA_WAS_RUNNING=0
    LLAMA_WAS_RUNNING=0
    [ -n "$(strata_pids)" ] && STRATA_WAS_RUNNING=1
    llama_running && LLAMA_WAS_RUNNING=1
    SERVICES_STOPPED=1

    if [ "$LLAMA_WAS_RUNNING" -eq 1 ]; then
        echo "stopping llama-swap before book search"
        "$LLAMA_SWAP" stop 9>&- || return 1
    fi
    if [ "$STRATA_WAS_RUNNING" -eq 1 ]; then
        echo "stopping Strata server before book search"
        for pid in $(strata_pids); do
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || strata_running || true
        done
        for _ in $(seq 1 30); do
            strata_running || return 0
            sleep 2
        done
        echo "Strata server did not stop within 60 seconds" >&2
        for pid in $(strata_pids); do
            pkill -KILL -P "$pid" 2>/dev/null || true
            kill -KILL "$pid" 2>/dev/null || true
        done
    fi
}

restore_model_services() {
    trap '' INT TERM HUP
    [ "$SERVICES_STOPPED" -eq 1 ] || return 0
    if [ "$STRATA_WAS_RUNNING" -eq 1 ]; then
        echo "restarting Strata server"
        for _ in $(seq 1 60); do   # at least as long as stop_model_services' own 60s stop budget
            strata_running || break
            sleep 1
        done
        if strata_running; then
            echo "warning: old Strata process is still alive after waiting 60s; not starting a second copy. Restart it by hand once it exits." >&2
        else
            # Subshell resets INT/TERM/HUP to defaults and closes the lock fd 9
            # before exec, so the long-lived daemon neither inherits this
            # script's ignored signals nor holds the book-build lock open.
            (
                trap - INT TERM HUP
                exec 9>&-
                exec nohup "$STRATA_START" >>/tmp/strata-orca-run.log 2>&1 </dev/null
            ) &
            sleep 3
            strata_running || echo "warning: Strata did not come back; see /tmp/strata-orca-run.log" >&2
        fi
    fi
    if [ "$LLAMA_WAS_RUNNING" -eq 1 ]; then
        echo "restarting llama-swap"
        "$LLAMA_SWAP" start 9>&- || echo "warning: could not restart llama-swap" >&2
    fi
    SERVICES_STOPPED=0
}

# shortcut: the EXIT trap below does not run on SIGKILL or a crash, so a
# service stopped for this run stays stopped if that happens; restart it by hand.
trap restore_model_services EXIT

preflight_interrupt() {
    trap '' INT TERM HUP
    echo "run.sh: interrupted during GPU preflight; leaving the previous export untouched" >&2
    exit 130
}
trap preflight_interrupt INT TERM HUP
is_positive_int() { case "$1" in ''|*[!0-9]*) return 1 ;; esac; [ "$((10#$1))" -ge 1 ]; }
is_positive_int "$SEARCH_SECONDS" || { echo "usage: bash $0 [SEARCH_SECONDS] [GPU_COUNT]  (SEARCH_SECONDS must be a positive integer)" >&2; exit 2; }
is_positive_int "$GPU_COUNT" || { echo "GPU_COUNT must be a positive integer, got '$GPU_COUNT'" >&2; exit 2; }
GPU_COUNT=$((10#$GPU_COUNT))   # '010' is ten, not octal eight
SEARCH_SECONDS=$((10#$SEARCH_SECONDS))
MIN_FREE_MIB="${MIN_FREE_MIB:-4000}"
case "$MIN_FREE_MIB" in ''|*[!0-9]*) echo "MIN_FREE_MIB must be a non-negative integer, got '$MIN_FREE_MIB'" >&2; exit 2 ;; esac
SIGINT_GRACE_STEPS="${SIGINT_GRACE_STEPS:-200}"   # 5 s each
is_positive_int "$SIGINT_GRACE_STEPS" || { echo "SIGINT_GRACE_STEPS must be a positive integer, got '$SIGINT_GRACE_STEPS'" >&2; exit 2; }
gpu_preflight() {
    for gpu in $(seq 0 $((GPU_COUNT - 1))); do
        free=$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
        case "$free" in
            ''|*[!0-9]*) echo "GPU $gpu: no usable reading from nvidia-smi (is there a GPU $gpu?)" >&2; return 2 ;;
        esac
        if [ "$free" -lt "$MIN_FREE_MIB" ]; then
            echo "GPU $gpu has $free MiB free (need $MIN_FREE_MIB)" >&2
            return 1
        fi
    done
    return 0
}

preflight_rc=0
gpu_preflight || preflight_rc=$?
if [ "$preflight_rc" -eq 1 ]; then
    if [ "${PREFLIGHT_ONLY:-0}" = 1 ]; then
        die 1 "GPU preflight failed (PREFLIGHT_ONLY=1 does not stop model services)"
    fi
    echo "insufficient GPU memory; stopping model services before retrying"
    stop_model_services || die 1 "could not stop the model services safely"
    for _ in $(seq 1 12); do
        gpu_preflight && break
        sleep 5
    done
    gpu_preflight || die 1 "GPU preflight still failed after stopping the model services"
elif [ "$preflight_rc" -ne 0 ]; then
    die 1 "GPU preflight failed"
fi
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
# Ctrl-C during preflight or service shutdown still runs the restore trap.
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
