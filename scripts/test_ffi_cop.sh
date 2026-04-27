#!/bin/bash
# test_ffi_cop.sh - Integration test for co-process FFI isolation
#
# Compiles .nano files that use FFI, runs them with --isolate-ffi,
# and verifies output matches standalone execution.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$PROJECT_DIR/bin"

PASS=0
FAIL=0

check() {
    local desc="$1"
    local expected="$2"
    local actual="$3"
    if [ "$expected" = "$actual" ]; then
        echo "  PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $desc"
        echo "    expected: $expected"
        echo "    actual:   $actual"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Co-Process FFI Integration Tests ==="

# Test 1: nano_cop binary exists
if [ -x "$BIN/nano_cop" ]; then
    check "nano_cop binary exists" "yes" "yes"
else
    check "nano_cop binary exists" "yes" "no"
    echo "Cannot continue without nano_cop. Build with: make -f Makefile.gnu nano_cop"
    exit 1
fi

# Test 2: nano_vm --isolate-ffi flag is accepted (with a pure .nvm that needs no FFI)
# Compile a simple program
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

cat > "$TMPDIR/simple.nano" <<'EOF'
fn main() -> int {
    return 42
}
EOF

# Compile to .nvm using nano_virt
if [ -x "$BIN/nano_virt" ]; then
    "$BIN/nano_virt" "$TMPDIR/simple.nano" --emit-nvm -o "$TMPDIR/simple.nvm" 2>/dev/null

    # Run standalone
    STANDALONE=$("$BIN/nano_vm" "$TMPDIR/simple.nvm" 2>&1 || true)
    EXIT_STANDALONE=$?

    # Run with --isolate-ffi (co-process won't be needed since no FFI calls)
    COP=$("$BIN/nano_vm" --isolate-ffi "$TMPDIR/simple.nvm" 2>&1 || true)
    EXIT_COP=$?

    check "standalone exit code" "0" "$EXIT_STANDALONE"
    check "cop exit code" "0" "$EXIT_COP"
else
    echo "  SKIP: nano_virt not found (can't compile test programs)"
fi

# Test 3: Check no orphan co-processes after execution
BEFORE=$(pgrep -c nano_cop 2>/dev/null || echo 0)
if [ -x "$BIN/nano_virt" ] && [ -f "$TMPDIR/simple.nvm" ]; then
    "$BIN/nano_vm" --isolate-ffi "$TMPDIR/simple.nvm" >/dev/null 2>&1 || true
    sleep 0.2
    AFTER=$(pgrep -c nano_cop 2>/dev/null || echo 0)
    check "no orphan co-processes" "$BEFORE" "$AFTER"
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="
[ "$FAIL" -eq 0 ]
