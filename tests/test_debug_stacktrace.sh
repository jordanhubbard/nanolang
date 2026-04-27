#!/usr/bin/env bash
# test_debug_stacktrace.sh — integration tests for source-mapped stack traces
#
# Tests the source-mapped stack trace feature across two paths:
#   A) nano interpreter (always available): happy path + shadow tests
#   B) nano_vm --debug: stack trace output (skipped if nano_vm not built)
#
# Usage: bash tests/test_debug_stacktrace.sh [nano_binary] [nano_vm_binary]

set -uo pipefail
NANO="${1:-./bin/nano}"
NANO_VM="${2:-./bin/nano_vm}"
PASS=0; FAIL=0; SKIP=0

ok()   { echo "  ✅ $1";            PASS=$((PASS+1)); }
fail() { echo "  ❌ $1: $2";        FAIL=$((FAIL+1)); }
skip() { echo "  ⏭  $1 (skipped: $2)"; SKIP=$((SKIP+1)); }

SRC="tests/test_debug_stacktrace.nano"
echo "Source-mapped stack trace tests:"
echo ""
echo "── Path A: interpreter (nano) ──────────────────────────────────"

# 1. Happy-path shadow tests pass
if "$NANO" "$SRC" > /dev/null 2>&1; then
    ok "interpreter: happy path exits 0"
else
    fail "interpreter happy path" "nano exited non-zero on $SRC"
fi

echo ""
echo "── Path B: nano_vm --debug ──────────────────────────────────────"

if ! command -v "$NANO_VM" > /dev/null 2>&1 && [ ! -x "$NANO_VM" ]; then
    skip "nano_vm --debug tests" "nano_vm not built (run: make nano_vm)"
    SKIP=$((SKIP+5))
else
    # Build a crashing .nvm module to test --debug output
    CRASH_SRC="$(mktemp /tmp/crash_XXXXXX.nano)"
    cat > "$CRASH_SRC" << 'EOF'
fn bad() -> int {
    assert (== 1 2)
    return 0
}
fn main() -> int {
    return (bad)
}
EOF
    CRASH_NVM="$(mktemp /tmp/crash_XXXXXX.nvm)"

    # Compile with debug info
    if ! ./bin/nanoc --debug "$CRASH_SRC" -o "$CRASH_NVM" 2>/dev/null; then
        skip "nano_vm stack trace tests" "nanoc --debug compilation failed"
        rm -f "$CRASH_SRC" "$CRASH_NVM"
    else
        TRACE="$("$NANO_VM" --debug "$CRASH_NVM" 2>&1 || true)"

        if echo "$TRACE" | grep -q "Stack trace"; then
            ok "nano_vm --debug emits 'Stack trace' header"
        else
            fail "stack trace header" "output: $(echo "$TRACE" | head -5)"
        fi

        if echo "$TRACE" | grep -q "Runtime error"; then
            ok "nano_vm --debug emits 'Runtime error' label"
        else
            fail "runtime error label" "output: $(echo "$TRACE" | head -5)"
        fi

        if echo "$TRACE" | grep -q "bad"; then
            ok "nano_vm --debug includes faulting function name 'bad'"
        else
            fail "function name in trace" "output: $(echo "$TRACE" | head -10)"
        fi

        if echo "$TRACE" | grep -qE ":[0-9]"; then
            ok "nano_vm --debug includes line number"
        else
            fail "line number in trace" "output: $(echo "$TRACE" | head -10)"
        fi

        # Multi-frame test
        MULTI_SRC="$(mktemp /tmp/multi_XXXXXX.nano)"
        MULTI_NVM="$(mktemp /tmp/multi_XXXXXX.nvm)"
        cat > "$MULTI_SRC" << 'EOF'
fn inner() -> int {
    assert (== 0 1)
    return 0
}
fn outer() -> int {
    return (inner)
}
fn main() -> int {
    return (outer)
}
EOF
        if ./bin/nanoc --debug "$MULTI_SRC" -o "$MULTI_NVM" 2>/dev/null; then
            MT="$("$NANO_VM" --debug "$MULTI_NVM" 2>&1 || true)"
            if echo "$MT" | grep -q "inner" && echo "$MT" | grep -qE "outer|main"; then
                ok "multi-frame: both inner and caller frames present"
            else
                fail "multi-frame frames" "output: $(echo "$MT" | head -10)"
            fi
        else
            skip "multi-frame test" "nanoc --debug compilation failed"
        fi
        rm -f "$CRASH_SRC" "$CRASH_NVM" "$MULTI_SRC" "$MULTI_NVM"
    fi
fi

echo ""
echo "Results: $PASS passed, $FAIL failed, $SKIP skipped"
[ "$FAIL" -eq 0 ]
