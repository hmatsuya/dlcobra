#!/usr/bin/env bash
# Switch between book building and llama serving as the Windows host appears/disappears.
set -u

PATH=/usr/bin:/bin
export PATH

REPO=${WINDOWS_BOOK_REPO:-/home/hmatsuya/workspace/Shogi/dlcobra}
CTL=${WINDOWS_BOOK_CTL:-$REPO/dlshogi/book/bookctl.sh}
PING=${WINDOWS_BOOK_PING:-/usr/bin/ping}
HOST=192.168.1.110
STATE_DIR=${WINDOWS_BOOK_STATE_DIR:-$REPO/output/.puct-book}
STATE_FILE=$STATE_DIR/windows-host.state
PENDING_FILE=$STATE_DIR/windows-host.pending
LOG_FILE=${WINDOWS_BOOK_LOG:-$REPO/output/windows-book-watch.log}
LOCK_FILE=$STATE_DIR/windows-book-watch.lock
DATE=/usr/bin/date
GRACE_SECONDS=600

if ! mkdir -p -- "$STATE_DIR" || ! chmod 700 -- "$STATE_DIR"; then
  exit 1
fi
exec 9>"$LOCK_FILE" || exit 1
/usr/bin/flock -n 9 || exit 0

log() {
  local stamp
  stamp=$($DATE -Is 2>/dev/null || printf '%s' unknown)
  printf '%s %s\n' "$stamp" "$*" >>"$LOG_FILE"
}

write_state() {
  local value=$1 tmp
  tmp=$(mktemp -- "$STATE_FILE.tmp.XXXXXX") || return 1
  if ! printf '%s\n' "$value" >"$tmp" || ! mv -f -- "$tmp" "$STATE_FILE"; then
    rm -f -- "$tmp"
    return 1
  fi
}

write_pending() {
  local value=$1 tmp
  tmp=$(mktemp -- "$PENDING_FILE.tmp.XXXXXX") || return 1
  if ! printf '%s\n' "$value" >"$tmp" || ! mv -f -- "$tmp" "$PENDING_FILE"; then
    rm -f -- "$tmp"
    return 1
  fi
}

# An absent or malformed state is unknown: it never switches to llama until a down state is recorded.
previous=
if [ -r "$STATE_FILE" ]; then
  state=$(cat -- "$STATE_FILE" 2>/dev/null) || state=
  case "$state" in
    up|down) previous=$state ;;
  esac
fi

reachable=up
"$PING" -c 1 -W 1 "$HOST" >/dev/null 2>&1 || reachable=down

now=$($DATE +%s 2>/dev/null) || {
  log "failure: could not read current time"
  exit 1
}
case "$now" in
  ''|*[!0-9]*)
    log "failure: invalid current time '$now'"
    exit 1
    ;;
esac

pending=
if [ -r "$PENDING_FILE" ]; then
  pending_value=$(cat -- "$PENDING_FILE" 2>/dev/null) || pending_value=
  case "$pending_value" in
    ''|*[!0-9]*) ;;
    *)
      [ "${#pending_value}" -le 12 ] && pending=$((10#$pending_value))
      ;;
  esac
fi

if [ "$reachable" = up ]; then
  if [ -e "$PENDING_FILE" ]; then
    log "Windows host $HOST returned during grace; canceling pending book transition"
    if ! rm -f -- "$PENDING_FILE"; then
      log "failure: could not cancel pending book transition"
      exit 1
    fi
  fi

  if [ "$previous" = down ]; then
    log "Windows host $HOST is up; switching to llama mode"
    if ! "$CTL" stop >>"$LOG_FILE" 2>&1; then
      log "failure: puct-bookctl stop failed; retaining down state"
      exit 1
    fi
    if ! "$CTL" llama >>"$LOG_FILE" 2>&1; then
      log "failure: puct-bookctl llama failed; retaining down state"
      exit 1
    fi
    if ! write_state up; then
      log "failure: could not record up state after llama switch"
      exit 1
    fi
  elif [ "$previous" != up ]; then
    if ! write_state up; then
      log "failure: could not record initial up state"
      exit 1
    fi
  fi
  exit 0
fi

# A recorded down state means book mode is already active.  Remove only a stale
# pending marker left after a state write succeeded; never start book twice.
if [ "$previous" = down ]; then
  if [ -e "$PENDING_FILE" ] && ! rm -f -- "$PENDING_FILE"; then
    log "failure: could not clear stale pending down timestamp"
    exit 1
  fi
  exit 0
fi

# A down host gets a fresh grace period unless one is already pending.
if [ -z "$pending" ]; then
  if write_pending "$now"; then
    log "Windows host $HOST is down; pending book transition for ${GRACE_SECONDS}s"
  else
    log "failure: could not record pending down timestamp"
    exit 1
  fi
  exit 0
fi

elapsed=$((now - pending))
if [ "$elapsed" -lt "$GRACE_SECONDS" ]; then
  exit 0
fi

log "Windows host $HOST remained down for ${GRACE_SECONDS}s; starting book mode"
if ! "$CTL" book >>"$LOG_FILE" 2>&1; then
  log "failure: puct-bookctl book failed; retaining pending down transition"
  exit 1
fi
if ! write_state down; then
  log "failure: book mode started but could not record down state"
  exit 1
fi
if ! rm -f -- "$PENDING_FILE"; then
  log "failure: could not clear pending down timestamp"
  exit 1
fi
log "book mode started; recorded down state"
exit 0
