#!/usr/bin/env bash
#
# test_nanovm_daemon.sh - Integration test for nano_vmd daemon
#
# Compiles .nano test files to .nvm, runs them both standalone and via the
# daemon, and verifies identical output.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BIN="$PROJECT_DIR/bin"
TMPDIR=$(mktemp -d /tmp/nanovm_daemon_test.XXXXXX)
DAEMON_PID=""

cleanup() {
    if [ -n "$DAEMON_PID" ] && kill -0 "$DAEMON_PID" 2>/dev/null; then
        kill "$DAEMON_PID" 2>/dev/null || true
        wait "$DAEMON_PID" 2>/dev/null || true
    fi
    rm -f "/tmp/nanolang_vm_$(id -u).sock" "/tmp/nanolang_vm_$(id -u).pid" 2>/dev/null || true
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

# Ensure binaries exist
for b in nano_virt nano_vm nano_vmd; do
    if [ ! -x "$BIN/$b" ]; then
        echo "ERROR: $BIN/$b not found. Run: make -f Makefile.gnu nano_virt nano_vm nano_vmd"
        exit 1
    fi
done

# Kill any stale daemon
pkill -f nano_vmd 2>/dev/null || true
sleep 0.3
rm -f "/tmp/nanolang_vm_$(id -u).sock" "/tmp/nanolang_vm_$(id -u).pid" 2>/dev/null || true

# Test files: use selfhost tests (pure NanoLang, no FFI needed)
TEST_FILES=(
    tests/selfhost/test_arithmetic_ops.nano
    tests/selfhost/test_comparison_ops.nano
    tests/selfhost/test_function_calls.nano
    tests/selfhost/test_if_else.nano
    tests/selfhost/test_let_set.nano
    tests/selfhost/test_logical_ops.nano
    tests/selfhost/test_recursion.nano
    tests/selfhost/test_while_loops.nano
)

passed=0
failed=0
skipped=0

echo "=== NanoVM Daemon Integration Test ==="
echo ""

# Phase 1: Compile all test files to .nvm
echo "--- Compiling test files ---"
compiled=()
for src in "${TEST_FILES[@]}"; do
    base=$(basename "$src" .nano)
    nvm="$TMPDIR/${base}.nvm"
    if "$BIN/nano_virt" "$PROJECT_DIR/$src" -o "$nvm" 2>/dev/null; then
        compiled+=("$nvm:$base")
    else
        echo "  SKIP: $src (compile failed)"
        ((skipped++)) || true
    fi
done
echo "  Compiled ${#compiled[@]} files"
echo ""

if [ ${#compiled[@]} -eq 0 ]; then
    echo "ERROR: No test files compiled successfully"
    exit 1
fi

# Phase 2: Run standalone and capture output
echo "--- Running standalone tests ---"
for entry in "${compiled[@]}"; do
    nvm="${entry%%:*}"
    base="${entry##*:}"
    "$BIN/nano_vm" "$nvm" > "$TMPDIR/${base}.standalone.out" 2>&1 || true
done
echo "  Done"
echo ""

# Phase 3: Start daemon in foreground mode (stderr to log file)
echo "--- Starting daemon ---"
"$BIN/nano_vmd" --foreground --no-timeout 2>"$TMPDIR/daemon.log" &
DAEMON_PID=$!

# Wait for socket to appear
for i in $(seq 1 50); do
    if [ -S "/tmp/nanolang_vm_$(id -u).sock" ]; then
        break
    fi
    sleep 0.1
done

if ! kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "ERROR: Daemon failed to start"
    cat "$TMPDIR/daemon.log"
    exit 1
fi
echo "  Daemon running (pid $DAEMON_PID)"
echo ""

# Phase 4: Run via daemon and compare
echo "--- Running daemon tests ---"
for entry in "${compiled[@]}"; do
    nvm="${entry%%:*}"
    base="${entry##*:}"

    "$BIN/nano_vm" --daemon "$nvm" > "$TMPDIR/${base}.daemon.out" 2>&1 || true

    if diff -q "$TMPDIR/${base}.standalone.out" "$TMPDIR/${base}.daemon.out" >/dev/null 2>&1; then
        echo "  PASS: $base"
        ((passed++)) || true
    else
        echo "  FAIL: $base (output differs)"
        echo "    --- standalone ---"
        head -5 "$TMPDIR/${base}.standalone.out" | sed 's/^/    /'
        echo "    --- daemon ---"
        head -5 "$TMPDIR/${base}.daemon.out" | sed 's/^/    /'
        ((failed++)) || true
    fi
done

echo ""

# Phase 5: Check daemon is still alive (stress test)
if kill -0 "$DAEMON_PID" 2>/dev/null; then
    echo "Daemon survived all tests (OK)"
else
    echo "WARNING: Daemon died during test"
fi

echo ""
echo "=== Results: $passed passed, $failed failed, $skipped skipped ==="

if [ "$failed" -gt 0 ]; then
    exit 1
fi
echo "All daemon tests passed!"
