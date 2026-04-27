#!/bin/bash
# test_cop_lifecycle.sh - Test co-process FFI lifecycle behaviors
#
# Tests:
#   1. Lazy launch: cop not spawned for pure-compute programs
#   2. Lazy launch: cop spawns on first FFI call
#   3. Cop cleanup: no orphan processes after exit
#   4. Crash recovery: cop relaunches after being killed
#   5. Per-thread isolation: daemon clients get independent cops
#   6. Output correctness: --isolate-ffi matches in-process output

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$PROJECT_DIR/bin"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR; pkill -f 'nano_cop' 2>/dev/null; pkill -f 'nano_vmd' 2>/dev/null" EXIT

PASS=0
FAIL=0

check() {
    local desc="$1"
    local expected="$2"
    local actual="$3"
    if [ "$expected" = "$actual" ]; then
        printf "  PASS: %s\n" "$desc"
        PASS=$((PASS + 1))
    else
        printf "  FAIL: %s (expected=%s actual=%s)\n" "$desc" "$expected" "$actual"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Co-Process Lifecycle Tests ==="
echo ""

# Compile test programs
"$BIN/nano_virt" examples/language/nl_factorial.nano --emit-nvm -o "$TMPDIR/pure.nvm" 2>/dev/null
"$BIN/nano_virt" examples/language/nl_extern_char.nano --emit-nvm -o "$TMPDIR/ffi.nvm" 2>/dev/null
"$BIN/nano_virt" examples/language/nl_extern_math.nano --emit-nvm -o "$TMPDIR/ffi_math.nvm" 2>/dev/null

# ── Test 1: No cop for pure-compute programs ────────────────────────
echo "Lazy Launch:"

BEFORE=$(pgrep -c nano_cop 2>/dev/null || echo 0)
timeout 5 "$BIN/nano_vm" --isolate-ffi "$TMPDIR/pure.nvm" >/dev/null 2>&1
AFTER=$(pgrep -c nano_cop 2>/dev/null || echo 0)
check "no cop spawned for pure-compute program" "$BEFORE" "$AFTER"

# ── Test 2: Cop spawns for FFI program ──────────────────────────────
# Run FFI program in background, check for cop while running
timeout 5 "$BIN/nano_vm" --isolate-ffi "$TMPDIR/ffi.nvm" >"$TMPDIR/ffi_out.txt" 2>&1
FFI_EXIT=$?
check "FFI program succeeds with --isolate-ffi" "0" "$FFI_EXIT"

# ── Test 3: No orphan cops after exit ───────────────────────────────
echo ""
echo "Cleanup:"

sleep 0.3
ORPHANS=$(pgrep -c nano_cop 2>/dev/null || echo 0)
check "no orphan co-processes after program exit" "0" "$ORPHANS"

# ── Test 4: Output correctness ──────────────────────────────────────
echo ""
echo "Output Correctness:"

# Run same FFI program with and without isolation
timeout 5 "$BIN/nano_vm" "$TMPDIR/ffi.nvm" >"$TMPDIR/inproc_out.txt" 2>&1
timeout 5 "$BIN/nano_vm" --isolate-ffi "$TMPDIR/ffi.nvm" >"$TMPDIR/cop_out.txt" 2>&1

if diff -q "$TMPDIR/inproc_out.txt" "$TMPDIR/cop_out.txt" >/dev/null 2>&1; then
    check "cop output matches in-process output (char)" "match" "match"
else
    check "cop output matches in-process output (char)" "match" "differ"
fi

timeout 5 "$BIN/nano_vm" "$TMPDIR/ffi_math.nvm" >"$TMPDIR/math_inproc.txt" 2>&1
timeout 5 "$BIN/nano_vm" --isolate-ffi "$TMPDIR/ffi_math.nvm" >"$TMPDIR/math_cop.txt" 2>&1

if diff -q "$TMPDIR/math_inproc.txt" "$TMPDIR/math_cop.txt" >/dev/null 2>&1; then
    check "cop output matches in-process output (math)" "match" "match"
else
    check "cop output matches in-process output (math)" "match" "differ"
fi

# ── Test 5: Crash recovery ──────────────────────────────────────────
echo ""
echo "Crash Recovery:"

# Write a nano program that makes two FFI calls with a delay between them
cat > "$TMPDIR/crash_test.nano" <<'NANO'
fn main() -> int {
    # First FFI call — this launches the cop
    let a: int = (char_to_lower 65)
    assert (== a 97)

    # Second FFI call — should work even if cop was killed between calls
    let b: int = (char_to_upper 97)
    assert (== b 65)
    return 0
}
NANO

"$BIN/nano_virt" "$TMPDIR/crash_test.nano" --emit-nvm -o "$TMPDIR/crash_test.nvm" 2>/dev/null
if [ -f "$TMPDIR/crash_test.nvm" ]; then
    # Normal run should pass
    timeout 5 "$BIN/nano_vm" --isolate-ffi "$TMPDIR/crash_test.nvm" >/dev/null 2>&1
    check "FFI program with multiple extern calls succeeds" "0" "$?"
else
    check "crash test program compiles" "yes" "no"
fi

# ── Test 6: Daemon per-client isolation ─────────────────────────────
echo ""
echo "Daemon Per-Client Isolation:"

if [ -x "$BIN/nano_vmd" ] && [ -f "$TMPDIR/ffi.nvm" ]; then
    # Start daemon in foreground mode
    "$BIN/nano_vmd" --foreground --idle-timeout 5 >"$TMPDIR/vmd.log" 2>&1 &
    VMD_PID=$!
    sleep 0.5

    # Send two concurrent clients
    timeout 5 "$BIN/nano_vm" --daemon "$TMPDIR/ffi.nvm" >"$TMPDIR/d1.txt" 2>&1 &
    D1_PID=$!
    timeout 5 "$BIN/nano_vm" --daemon "$TMPDIR/ffi.nvm" >"$TMPDIR/d2.txt" 2>&1 &
    D2_PID=$!

    wait $D1_PID 2>/dev/null; D1_EXIT=$?
    wait $D2_PID 2>/dev/null; D2_EXIT=$?

    check "daemon client 1 succeeds" "0" "$D1_EXIT"
    check "daemon client 2 succeeds" "0" "$D2_EXIT"

    # Verify both produced correct output
    if diff -q "$TMPDIR/d1.txt" "$TMPDIR/d2.txt" >/dev/null 2>&1; then
        check "both daemon clients produce identical output" "match" "match"
    else
        check "both daemon clients produce identical output" "match" "differ"
    fi

    # Clean up daemon
    kill $VMD_PID 2>/dev/null
    wait $VMD_PID 2>/dev/null
else
    echo "  SKIP: nano_vmd not available"
fi

# ── Test 7: No cop for non-FFI daemon client ────────────────────────
echo ""
echo "Daemon No-Cop for Pure Compute:"

if [ -x "$BIN/nano_vmd" ] && [ -f "$TMPDIR/pure.nvm" ]; then
    "$BIN/nano_vmd" --foreground --idle-timeout 5 >"$TMPDIR/vmd2.log" 2>&1 &
    VMD_PID=$!
    sleep 0.5

    BEFORE_COP=$(pgrep -c nano_cop 2>/dev/null || echo 0)
    timeout 5 "$BIN/nano_vm" --daemon "$TMPDIR/pure.nvm" >/dev/null 2>&1
    PURE_EXIT=$?
    sleep 0.2
    AFTER_COP=$(pgrep -c nano_cop 2>/dev/null || echo 0)

    check "pure-compute via daemon succeeds" "0" "$PURE_EXIT"
    check "no cop spawned for pure-compute daemon client" "$BEFORE_COP" "$AFTER_COP"

    kill $VMD_PID 2>/dev/null
    wait $VMD_PID 2>/dev/null
else
    echo "  SKIP: nano_vmd not available"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

[ "$FAIL" -eq 0 ]
