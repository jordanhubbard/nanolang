#!/bin/bash
# test_cop_lifecycle.sh - Test co-process FFI lifecycle behaviors without
# inspecting or killing processes owned by another invocation.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BIN="${NANO_COP_LIFECYCLE_BIN_DIR:-$PROJECT_DIR/bin}"
PROBE="${NANO_COP_LIFECYCLE_PROBE:-$PROJECT_DIR/obj/test_cop_lifecycle}"
TMP_BASE="${TMPDIR:-/tmp}"
TMP_BASE="${TMP_BASE%/}"
WORK_DIR=$(mktemp -d "$TMP_BASE/nanolang-cop-lifecycle.XXXXXX") || exit 1
OWNED_PIDS=""
PASS=0
FAIL=0

track_pid() {
    local pid="$1"
    OWNED_PIDS="$OWNED_PIDS $pid"
    if [ -n "${NANO_COP_LIFECYCLE_PID_LOG:-}" ]; then
        printf '%s\n' "$pid" >> "$NANO_COP_LIFECYCLE_PID_LOG"
    fi
}

untrack_pid() {
    local target="$1"
    local pid
    local remaining=""
    for pid in $OWNED_PIDS; do
        if [ "$pid" != "$target" ]; then remaining="$remaining $pid"; fi
    done
    OWNED_PIDS="$remaining"
}

stop_owned() {
    local pid="$1"
    local attempt
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        for attempt in 1 2 3 4 5 6 7 8 9 10; do
            if ! kill -0 "$pid" 2>/dev/null; then break; fi
            sleep 0.02
        done
        if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid" 2>/dev/null || true; fi
    fi
    wait "$pid" 2>/dev/null || true
    untrack_pid "$pid"
}

wait_owned() {
    local pid="$1"
    local result
    if wait "$pid"; then result=0; else result=$?; fi
    untrack_pid "$pid"
    return "$result"
}

cleanup() {
    local result=$?
    local pid
    trap - EXIT INT TERM HUP
    for pid in $OWNED_PIDS; do stop_owned "$pid"; done
    case "$WORK_DIR" in
        "$TMP_BASE"/nanolang-cop-lifecycle.*) rm -rf -- "$WORK_DIR" ;;
        *) printf 'I refuse to remove an unexpected lifecycle path: %s\n' "$WORK_DIR" >&2 ;;
    esac
    exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

check() {
    local desc="$1"
    local expected="$2"
    local actual="$3"
    if [ "$expected" = "$actual" ]; then
        printf '  PASS: %s\n' "$desc"
        PASS=$((PASS + 1))
    else
        printf '  FAIL: %s (expected=%s actual=%s)\n' "$desc" "$expected" "$actual"
        FAIL=$((FAIL + 1))
    fi
}

run_phase() {
    local phase="$1"
    shift
    if [ "${NANO_COP_LIFECYCLE_INJECT_FAIL:-}" = "$phase" ]; then
        printf 'I injected the %s failure.\n' "$phase" >&2
        return 97
    fi
    "$@"
}

run_bounded() {
    local seconds="$1"
    shift
    perl -e '$seconds = shift @ARGV; alarm $seconds; exec @ARGV; die "I cannot execute @ARGV: $!\n"' \
        "$seconds" "$@"
}

require_tool() {
    if [ ! -x "$1" ]; then
        printf 'I require an executable lifecycle prerequisite: %s\n' "$1" >&2
        exit 1
    fi
}

compile_program() {
    local phase="$1"
    local source="$2"
    local output="$3"
    local log="$4"
    if run_phase "$phase" "$BIN/nano_virt" "$source" --emit-nvm -o "$output" >"$log" 2>&1; then
        if [ -s "$output" ]; then return 0; fi
        printf 'I did not receive the %s artifact.\n' "$phase" >&2
    else
        printf 'I could not complete %s.\n' "$phase" >&2
    fi
    sed -n '1,160p' "$log" >&2
    return 1
}

start_daemon() {
    local phase="$1"
    local socket="$2"
    local log="$3"
    local attempt
    export NANOVMD_SOCKET="$socket"
    if [ "${NANO_COP_LIFECYCLE_INJECT_FAIL:-}" = "$phase" ]; then
        printf 'I injected the %s failure.\n' "$phase" >&2
        return 97
    fi
    "$BIN/nano_vmd" --foreground --idle-timeout 30 >"$log" 2>&1 &
    VMD_PID=$!
    track_pid "$VMD_PID"
    for attempt in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25; do
        if [ -S "$socket" ] && kill -0 "$VMD_PID" 2>/dev/null; then return 0; fi
        if ! kill -0 "$VMD_PID" 2>/dev/null; then break; fi
        sleep 0.02
    done
    printf 'I could not start my private daemon for %s.\n' "$phase" >&2
    sed -n '1,160p' "$log" >&2
    return 1
}

for tool in "$BIN/nano_virt" "$BIN/nano_vm" "$BIN/nano_vmd" "$PROBE"; do
    require_tool "$tool"
done

printf '=== Co-Process Lifecycle Tests ===\n\n'

if ! compile_program compile-pure examples/language/nl_factorial.nano \
        "$WORK_DIR/pure.nvm" "$WORK_DIR/compile-pure.log" ||
   ! compile_program compile-ffi examples/language/nl_extern_char.nano \
        "$WORK_DIR/ffi.nvm" "$WORK_DIR/compile-ffi.log" ||
   ! compile_program compile-math examples/language/nl_extern_math.nano \
        "$WORK_DIR/ffi_math.nvm" "$WORK_DIR/compile-math.log"; then
    printf '\n=== Results: %d passed, %d failed ===\n' "$PASS" "$((FAIL + 1))"
    exit 1
fi

printf 'Owned worker lifecycle:\n'
if run_phase owned-worker-probe run_bounded 10 "$PROBE" >"$WORK_DIR/probe.log" 2>&1; then
    check 'lazy launch, stable identity, crash recovery and owned reap' 0 0
else
    result=$?
    sed -n '1,160p' "$WORK_DIR/probe.log" >&2
    check 'lazy launch, stable identity, crash recovery and owned reap' 0 "$result"
fi

printf '\nDirect execution:\n'
if run_phase pure-execution run_bounded 5 "$BIN/nano_vm" --isolate-ffi \
        "$WORK_DIR/pure.nvm" >"$WORK_DIR/pure.out" 2>"$WORK_DIR/pure.err"; then
    check 'pure program succeeds with isolation requested' 0 0
else
    check 'pure program succeeds with isolation requested' 0 "$?"
fi
if run_phase ffi-execution run_bounded 5 "$BIN/nano_vm" --isolate-ffi \
        "$WORK_DIR/ffi.nvm" >"$WORK_DIR/ffi.out" 2>"$WORK_DIR/ffi.err"; then
    check 'FFI program succeeds with isolation requested' 0 0
else
    check 'FFI program succeeds with isolation requested' 0 "$?"
fi

printf '\nOutput parity:\n'
for kind in char math; do
    if [ "$kind" = char ]; then artifact="$WORK_DIR/ffi.nvm"; else artifact="$WORK_DIR/ffi_math.nvm"; fi
    if run_phase "${kind}-inproc" run_bounded 5 "$BIN/nano_vm" "$artifact" \
            >"$WORK_DIR/${kind}-inproc.out" 2>"$WORK_DIR/${kind}-inproc.err"; then
        inproc_status=0
    else
        inproc_status=$?
    fi
    if run_phase "${kind}-isolated" run_bounded 5 "$BIN/nano_vm" --isolate-ffi "$artifact" \
            >"$WORK_DIR/${kind}-isolated.out" 2>"$WORK_DIR/${kind}-isolated.err"; then
        isolated_status=0
    else
        isolated_status=$?
    fi
    check "$kind in-process execution succeeds" 0 "$inproc_status"
    check "$kind isolated execution succeeds" 0 "$isolated_status"
    if [ "$inproc_status" -eq 0 ] && [ "$isolated_status" -eq 0 ] &&
       diff -q "$WORK_DIR/${kind}-inproc.out" "$WORK_DIR/${kind}-isolated.out" >/dev/null; then
        check "$kind isolated output matches in-process output" match match
    else
        check "$kind isolated output matches in-process output" match differ
    fi
done

printf '\nPrivate daemon clients:\n'
if start_daemon daemon-start "$WORK_DIR/clients.sock" "$WORK_DIR/vmd-clients.log"; then
    run_phase daemon-client-1 run_bounded 5 env NANOVMD_SOCKET="$NANOVMD_SOCKET" \
        "$BIN/nano_vm" --daemon "$WORK_DIR/ffi.nvm" >"$WORK_DIR/client-1.out" 2>"$WORK_DIR/client-1.err" &
    D1_PID=$!
    track_pid "$D1_PID"
    run_phase daemon-client-2 run_bounded 5 env NANOVMD_SOCKET="$NANOVMD_SOCKET" \
        "$BIN/nano_vm" --daemon "$WORK_DIR/ffi.nvm" >"$WORK_DIR/client-2.out" 2>"$WORK_DIR/client-2.err" &
    D2_PID=$!
    track_pid "$D2_PID"
    if wait_owned "$D1_PID"; then D1_STATUS=0; else D1_STATUS=$?; fi
    if wait_owned "$D2_PID"; then D2_STATUS=0; else D2_STATUS=$?; fi
    check 'daemon client 1 succeeds' 0 "$D1_STATUS"
    check 'daemon client 2 succeeds' 0 "$D2_STATUS"
    if [ "$D1_STATUS" -eq 0 ] && [ "$D2_STATUS" -eq 0 ] &&
       diff -q "$WORK_DIR/client-1.out" "$WORK_DIR/client-2.out" >/dev/null; then
        check 'private daemon clients produce identical output' match match
    else
        check 'private daemon clients produce identical output' match differ
    fi
    stop_owned "$VMD_PID"
else
    check 'private daemon starts' 0 "$?"
fi

if start_daemon pure-daemon-start "$WORK_DIR/pure.sock" "$WORK_DIR/vmd-pure.log"; then
    if run_phase pure-daemon-client run_bounded 5 env NANOVMD_SOCKET="$NANOVMD_SOCKET" \
            "$BIN/nano_vm" --daemon "$WORK_DIR/pure.nvm" >"$WORK_DIR/pure-daemon.out" \
            2>"$WORK_DIR/pure-daemon.err"; then
        check 'pure program succeeds through private daemon' 0 0
    else
        check 'pure program succeeds through private daemon' 0 "$?"
    fi
    stop_owned "$VMD_PID"
else
    check 'private pure daemon starts' 0 "$?"
fi

printf '\n=== Results: %d passed, %d failed ===\n' "$PASS" "$FAIL"
[ "$FAIL" -eq 0 ]
