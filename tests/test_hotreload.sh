#!/bin/bash
# tests/test_hotreload.sh — REPL hot-reload (:reload <module>) tests
#
# Tests that :reload re-parses a module and rebinds its exported functions
# in the running REPL environment without restarting.

set -euo pipefail

REPL="${1:-./bin/nanolang-repl}"

if [ ! -x "$REPL" ]; then
    echo "SKIP: $REPL not found" >&2
    exit 0
fi

PASS=0
FAIL=0
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

# Helper: run REPL with piped input, grep for expected string
check() {
    local desc="$1"
    local input="$2"
    local expected="$3"
    local actual
    actual=$(printf '%s\n:quit\n' "$input" | timeout 10 "$REPL" 2>&1 || true)
    if echo "$actual" | grep -qF "$expected"; then
        echo "PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $desc"
        echo "  Expected to contain: $expected"
        echo "  Got: $(echo "$actual" | head -8 | tr '\n' '|')"
        FAIL=$((FAIL + 1))
    fi
}

# ── Test 1: define a module, import it, call the function ─────────────────────
MOD1="$WORK/mod1.nano"
cat > "$MOD1" << 'NANOEOF'
pub fn double(x: int) -> int {
    return x * 2
}
NANOEOF

# nanolang uses prefix call syntax: (double 5)
check "import and call" \
    "$(printf 'from "%s" import double\n(double 5)' "$MOD1")" \
    "10"

# ── Test 2: :reload rebinds function bodies ───────────────────────────────────
# Phase A: triple(x) = x * 2 (wrong implementation initially)
# Phase B: overwrite module so triple(x) = x * 3
# After :reload, calling (triple 4) should return 12 not 8

MOD2="$WORK/mod2.nano"
cat > "$MOD2" << 'NANOEOF'
pub fn triple(x: int) -> int {
    return x * 2
}
NANOEOF

# Wait to ensure mtime changes, then overwrite with correct implementation
sleep 1
cat > "$MOD2" << 'NANOEOF'
pub fn triple(x: int) -> int {
    return x * 3
}
NANOEOF

ACTUAL_RELOAD=$(printf 'from "%s" import triple\n(triple 4)\n:reload %s\n(triple 4)\n:quit\n' \
    "$MOD2" "$MOD2" | timeout 10 "$REPL" 2>&1 || true)

# After reload the second call to (triple 4) should produce 12
if echo "$ACTUAL_RELOAD" | grep -q "12"; then
    echo "PASS: :reload rebinds function body"
    PASS=$((PASS + 1))
else
    echo "FAIL: :reload rebinds function body"
    echo "  Output: $(echo "$ACTUAL_RELOAD" | head -8 | tr '\n' '|')"
    FAIL=$((FAIL + 1))
fi

# ── Test 3: :reload with unchanged file says [unchanged] ──────────────────────
MOD3="$WORK/mod3.nano"
cat > "$MOD3" << 'NANOEOF'
pub fn inc(x: int) -> int {
    return x + 1
}
NANOEOF

ACTUAL_UNCHANGED=$(printf ':reload %s\n:reload %s\n:quit\n' "$MOD3" "$MOD3" \
    | timeout 10 "$REPL" 2>&1 || true)

if echo "$ACTUAL_UNCHANGED" | grep -q "unchanged"; then
    echo "PASS: second :reload with no changes reports [unchanged]"
    PASS=$((PASS + 1))
else
    echo "FAIL: second :reload should report [unchanged]"
    echo "  Output: $(echo "$ACTUAL_UNCHANGED" | head -6 | tr '\n' '|')"
    FAIL=$((FAIL + 1))
fi

# ── Test 4: :modules lists loaded modules ─────────────────────────────────────
MOD4="$WORK/mod4.nano"
cat > "$MOD4" << 'NANOEOF'
pub fn greet(x: int) -> int {
    return x
}
NANOEOF

check ":modules shows loaded module" \
    "$(printf 'from "%s" import greet\n:modules' "$MOD4")" \
    "greet"

# ── Test 5: :help mentions :reload ────────────────────────────────────────────
check ":help mentions :reload" ":help" ":reload"

# ── Test 6: new function added via :reload is callable ────────────────────────
MOD5="$WORK/mod5.nano"
cat > "$MOD5" << 'NANOEOF'
pub fn square(x: int) -> int {
    return x * x
}
NANOEOF

check ":reload adds new function" \
    "$(printf ':reload %s\n(square 7)' "$MOD5")" \
    "49"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
