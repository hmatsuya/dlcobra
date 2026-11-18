#!/usr/bin/env bash
# Manage the PUCT book builder: start | stop | restart | status | llama | book | restore
set -u

SCRIPT_PATH=$(readlink -f -- "${BASH_SOURCE[0]}") || { echo "could not resolve script path" >&2; exit 1; }
REPO=$(cd -- "$(dirname -- "$SCRIPT_PATH")/../.." && pwd -P) || { echo "could not resolve project path" >&2; exit 1; }
RUN="$REPO/dlshogi/book/run.sh"
BOOK_LOCK_DIR="$REPO/output/.puct-book"
BOOK_LOCK_FILE="$BOOK_LOCK_DIR/book.lock"
NAME=puct-book
STATE_DIR="${PUCT_BOOK_STATE_DIR:-$BOOK_LOCK_DIR}"
PID_FILE="$STATE_DIR/$NAME.pid"
LOCK_FILE="$STATE_DIR/$NAME.ctl.lock"
LOG_FILE="${PUCT_BOOK_LOG:-$REPO/output/puct-book-run.log}"
RESTORE_FILE="$STATE_DIR/restore-services"
# run.sh can spend 1000s waiting for graceful search shutdown before restoration.
STOP_TIMEOUT="${PUCT_BOOK_STOP_TIMEOUT:-1200}"

if [ -L "$STATE_DIR" ] || { [ -e "$STATE_DIR" ] && [ ! -d "$STATE_DIR" ]; }; then
  echo "state path is not a directory: $STATE_DIR" >&2
  exit 1
fi
if [ ! -d "$STATE_DIR" ]; then
  (umask 077 && mkdir -p -- "$STATE_DIR") || { echo "could not create state directory: $STATE_DIR" >&2; exit 1; }
fi
chmod 700 -- "$STATE_DIR" || { echo "could not secure state directory: $STATE_DIR" >&2; exit 1; }
if [ "$STATE_DIR" != "$BOOK_LOCK_DIR" ]; then
  if [ -L "$BOOK_LOCK_DIR" ] || { [ -e "$BOOK_LOCK_DIR" ] && [ ! -d "$BOOK_LOCK_DIR" ]; }; then
    echo "book lock path is not a directory: $BOOK_LOCK_DIR" >&2
    exit 1
  fi
  if [ ! -d "$BOOK_LOCK_DIR" ]; then
    (umask 077 && mkdir -p -- "$BOOK_LOCK_DIR") || { echo "could not create book lock directory: $BOOK_LOCK_DIR" >&2; exit 1; }
  fi
  chmod 700 -- "$BOOK_LOCK_DIR" || { echo "could not secure book lock directory: $BOOK_LOCK_DIR" >&2; exit 1; }
fi

external_builder_running() {
  [ -e "$BOOK_LOCK_FILE" ] || return 1
  ! ( exec 8<"$BOOK_LOCK_FILE"; flock -n 8 )
}

resolve_service_paths() {
  [ -n "${HOME:-}" ] || { echo "HOME is unset; cannot resolve model service paths" >&2; return 1; }
  STRATA_CTL="$HOME/opt/strata-orca/strata.sh"
  LLAMA_SWAP="$HOME/opt/llama-swap/llama-swap.sh"
}

check_service_scripts() {
  [ -x "$STRATA_CTL" ] || { echo "service script missing or not executable: $STRATA_CTL" >&2; return 1; }
  [ -x "$LLAMA_SWAP" ] || { echo "service script missing or not executable: $LLAMA_SWAP" >&2; return 1; }
}

# Keep this identical to STRATA_PATTERN in run.sh. It does not match the sibling ~/opt/Strata install.
STRATA_PATTERN="/home/hmatsuya/opt/strata-orca/Strata/.venv/bin/python -m serve.server --engine strata"
strata_running() { pgrep -f -- "^$STRATA_PATTERN" >/dev/null 2>&1; }
llama_swap_running() {
  local containers
  containers=$(docker ps --format '{{.Names}}' 2>/dev/null) || return 2
  printf '%s\n' "$containers" | grep -qx llama-swap
}

write_restore_services() {
  local services= tmp llama_status
  [ -e "$RESTORE_FILE" ] && return 0
  if strata_running; then
    services+="strata\n"
  fi
  llama_swap_running
  llama_status=$?
  case "$llama_status" in
    0) services+="llama-swap\n" ;;
    1) ;;
    *) return 1 ;;
  esac
  tmp=$(mktemp "$RESTORE_FILE.tmp.XXXXXX") || return 1
  if ! printf '%b' "$services" >"$tmp" || ! mv -f -- "$tmp" "$RESTORE_FILE"; then
    rm -f -- "$tmp"
    return 1
  fi
}

restore_services() {
  local service restore_strata=0 restore_llama_swap=0
  if [ ! -e "$RESTORE_FILE" ]; then
    echo "no restore-services record exists; nothing to restore"
    return 0
  fi
  [ -r "$RESTORE_FILE" ] || { echo "restore-services record is not readable: $RESTORE_FILE" >&2; return 1; }
  while IFS= read -r service || [ -n "$service" ]; do
    case "$service" in
      '') ;;
      strata) restore_strata=1 ;;
      llama-swap) restore_llama_swap=1 ;;
      *) echo "invalid service in restore-services record: $service" >&2; return 1 ;;
    esac
  done <"$RESTORE_FILE"
  if [ "$restore_strata" -eq 0 ] && [ "$restore_llama_swap" -eq 0 ]; then
    rm -f -- "$RESTORE_FILE" || return 1
    echo "restore-services record was empty; nothing to restore"
    return 0
  fi
  resolve_service_paths || return 1
  check_service_scripts || return 1
  if [ "$restore_strata" -eq 1 ] && ! "$STRATA_CTL" start 9>&-; then
    echo "could not restore Strata; retaining restore-services record" >&2
    return 1
  fi
  if [ "$restore_llama_swap" -eq 1 ] && ! "$LLAMA_SWAP" start 9>&-; then
    echo "could not restore llama-swap; retaining restore-services record" >&2
    return 1
  fi
  rm -f -- "$RESTORE_FILE" || return 1
  echo "restored recorded model services"
}

is_positive_int() {
  case "$1" in ''|*[!0-9]*) return 1 ;; esac
  [ "$((10#$1))" -ge 1 ]
}

validate_book_args() {
  [ "$#" -le 2 ] || { echo "usage: $0 book [SEARCH_SECONDS] [GPU_COUNT]" >&2; return 2; }
  if [ "$#" -ge 1 ] && ! is_positive_int "$1"; then
    echo "SEARCH_SECONDS must be a positive integer, got '$1'" >&2
    return 2
  fi
  if [ "$#" -ge 2 ] && ! is_positive_int "$2"; then
    echo "GPU_COUNT must be a positive integer, got '$2'" >&2
    return 2
  fi
}
COMMAND=${1:-status}
exec 9>"$LOCK_FILE" || { echo "could not open controller lock: $LOCK_FILE" >&2; exit 1; }
if [ "$COMMAND" = status ]; then
  flock -n 9 || { echo "status unavailable: controller is busy" >&2; exit 1; }
else
  flock -x 9 || { echo "could not acquire controller lock: $LOCK_FILE" >&2; exit 1; }
fi

proc_start_time() {
  local pid=$1 stat
  [ -r "/proc/$pid/stat" ] || return 1
  IFS= read -r stat <"/proc/$pid/stat" || return 1
  stat=${stat##*') '}
  set -- $stat
  [ "$#" -ge 20 ] || return 1
  printf '%s\n' "${20}"
}

proc_cmdline_matches_run() {
  local pid=$1 arg
  [ -r "/proc/$pid/cmdline" ] || return 1
  while IFS= read -r -d '' arg; do
    [ "$arg" = "$RUN" ] && return 0
  done <"/proc/$pid/cmdline"
  return 1
}

read_metadata() {
  PID_FILE_PID=
  PID_FILE_START_TIME=
  [ -r "$PID_FILE" ] || return 1
  local line extra
  {
    IFS= read -r line || return 1
    IFS=$' \t' read -r PID_FILE_PID PID_FILE_START_TIME extra <<<"$line"
    [ -n "$PID_FILE_PID" ] && [ -n "$PID_FILE_START_TIME" ] && [ -z "$extra" ] || return 1
    IFS= read -r extra && return 1
  } <"$PID_FILE"
  case "$PID_FILE_PID" in ''|*[!0-9]*) return 1 ;; esac
  case "$PID_FILE_START_TIME" in ''|*[!0-9]*) return 1 ;; esac
  [ "$PID_FILE_PID" -gt 0 ] 2>/dev/null || return 1
}

write_metadata() {
  local pid=$1 start_time=$2 tmp
  tmp=$(mktemp "${PID_FILE}.tmp.XXXXXX") || return 1
  if ! printf '%s %s\n' "$pid" "$start_time" >"$tmp"; then
    rm -f -- "$tmp"
    return 1
  fi
  if ! mv -f -- "$tmp" "$PID_FILE"; then
    rm -f -- "$tmp"
    return 1
  fi
}

metadata_matches() {
  local pid=$1 start_time=$2
  read_metadata 2>/dev/null \
    && [ "$PID_FILE_PID" = "$pid" ] \
    && [ "$PID_FILE_START_TIME" = "$start_time" ]
}

remove_metadata_if_matches() {
  local pid=$1 start_time=$2
  metadata_matches "$pid" "$start_time" && rm -f -- "$PID_FILE"
}

process_matches() {
  local pid=${1:-} start_time=${2:-}
  [ -n "$pid" ] && [ -n "$start_time" ] || return 1
  [ "$(proc_start_time "$pid" 2>/dev/null || true)" = "$start_time" ] || return 1
  proc_cmdline_matches_run "$pid"
}

is_running() {
  local pid=${1:-} start_time=${2:-}
  process_matches "$pid" "$start_time" || return 1
  kill -0 "$pid" 2>/dev/null
}

process_has_exited() {
  local pid=$1 stat
  [ -e "/proc/$pid/stat" ] || return 0
  IFS= read -r stat <"/proc/$pid/stat" || return 1
  stat=${stat##*') '}
  set -- $stat
  [ "${1:-}" = Z ]
}

send_sigint() {
  local pid=$1 start_time=$2
  if ! process_matches "$pid" "$start_time"; then
    process_has_exited && return 0
    echo "could not send SIGINT to book builder (pid=$pid changed)" >&2
    return 1
  fi
  if kill -INT -- "-$pid" 2>/dev/null; then
    return 0
  fi
  process_has_exited && return 0
  echo "could not send SIGINT to book builder (pid=$pid)" >&2
  return 1
}

do_status() {
  if read_metadata 2>/dev/null && is_running "$PID_FILE_PID" "$PID_FILE_START_TIME"; then
    echo "RUNNING pid=$PID_FILE_PID log=$LOG_FILE"
    ps -o pid=,etime=,stat=,args= -p "$PID_FILE_PID"
    return 0
  fi
  if [ -e "$PID_FILE" ]; then
    echo "stopped (invalid or stale pid file)"
  else
    echo "stopped"
  fi
  return 1
}

do_start() {
  local pid start_time
  if read_metadata 2>/dev/null && is_running "$PID_FILE_PID" "$PID_FILE_START_TIME"; then
    echo "already running (pid=$PID_FILE_PID)"
    return 0
  fi
  [ -x "$RUN" ] || { echo "launcher missing or not executable: $RUN" >&2; return 1; }
  mkdir -p "$(dirname "$LOG_FILE")" || return 1
  echo "starting $NAME (log: $LOG_FILE)..."
  setsid "$RUN" "$@" 9>&- >>"$LOG_FILE" 2>&1 </dev/null &
  pid=$!
  start_time=$(proc_start_time "$pid" 2>/dev/null || true)
  if [ -z "$start_time" ] || ! write_metadata "$pid" "$start_time"; then
    echo "could not write process metadata; requesting startup process to stop" >&2
    if is_running "$pid" "$start_time"; then
      send_sigint "$pid" "$start_time"
    fi
    return 1
  fi
  sleep 1
  if is_running "$pid" "$start_time"; then
    echo "started (pid=$pid)"
    return 0
  fi
  echo "book builder exited during startup; last log lines:" >&2
  tail -20 "$LOG_FILE" >&2
  remove_metadata_if_matches "$pid" "$start_time"
  return 1
}

do_stop() {
  local pid start_time elapsed=0
  if [ ! -e "$PID_FILE" ]; then
    echo "not running."
    return 0
  fi
  if ! read_metadata 2>/dev/null; then
    echo "not running (invalid pid file)." >&2
    return 0
  fi
  pid=$PID_FILE_PID
  start_time=$PID_FILE_START_TIME
  if ! is_running "$pid" "$start_time"; then
    echo "not running (stale pid file)." >&2
    return 0
  fi
  case "$STOP_TIMEOUT" in ''|*[!0-9]*)
    echo "PUCT_BOOK_STOP_TIMEOUT must be a positive integer" >&2
    return 2
    ;;
  esac
  [ "$STOP_TIMEOUT" -gt 0 ] 2>/dev/null || {
    echo "PUCT_BOOK_STOP_TIMEOUT must be a positive integer" >&2
    return 2
  }
  echo "stopping $NAME (pid=$pid)..."
  is_running "$pid" "$start_time" || return 0
  send_sigint "$pid" "$start_time" || return 1
  while [ "$elapsed" -lt "$STOP_TIMEOUT" ]; do
    is_running "$pid" "$start_time" || { remove_metadata_if_matches "$pid" "$start_time"; echo "stopped."; return 0; }
    sleep 5
    elapsed=$((elapsed + 5))
  done
  if ! is_running "$pid" "$start_time"; then
    remove_metadata_if_matches "$pid" "$start_time"
    echo "stopped."
    return 0
  fi
  echo "book builder did not stop within ${STOP_TIMEOUT}s; leaving it running" >&2
  return 1
}

case "$COMMAND" in
  start) shift; do_start "$@" ;;
  stop) do_stop ;;
  restart) shift; do_stop && do_start "$@" ;;
  status) do_status ;;
  llama)
    [ "$#" -eq 1 ] || { echo "usage: $0 llama" >&2; exit 2; }
    if external_builder_running; then
      echo "a book builder holds the run lock; stop it with '$0 stop' before starting llama-swap" >&2
      exit 1
    fi
    resolve_service_paths || exit 1
    check_service_scripts || exit 1
    do_stop && "$STRATA_CTL" stop 9>&- && "$LLAMA_SWAP" start 9>&- && rm -f -- "$RESTORE_FILE"
    ;;
  book)
    shift
    validate_book_args "$@" || exit $?
    if external_builder_running; then
      echo "a book builder holds the run lock; stop it with '$0 stop' before starting another book build" >&2
      exit 1
    fi
    resolve_service_paths || exit 1
    check_service_scripts || exit 1
    [ -x "$RUN" ] || { echo "launcher missing or not executable: $RUN" >&2; exit 1; }
    write_restore_services || { echo "could not record model services for restoration" >&2; exit 1; }
    "$STRATA_CTL" stop 9>&- && "$LLAMA_SWAP" stop 9>&- && do_start "$@"
    ;;
  restore)
    [ "$#" -eq 1 ] || { echo "usage: $0 restore" >&2; exit 2; }
    if external_builder_running; then
      echo "a book builder holds the run lock; stop it with '$0 stop' before restoring model services" >&2
      exit 1
    fi
    restore_services
    ;;
  *) echo "usage: $0 {start|stop|restart|status|llama|book|restore} [SEARCH_SECONDS] [GPU_COUNT]"; exit 2 ;;
esac
