#!/usr/bin/env bash
# tests/test_proptest.sh — property-based test oracle smoke test
#
# Tests the --proptest flag of the nano interpreter:
#   1. All-passing properties: exit 0, output contains "PASS"
#   2. Known-broken property: exit 1, output contains "FAIL"
#
# Usage: ./tests/test_proptest.sh [--verbose]
#
# Requires: bin/nano must be built (make or make build)

set -euo pipefail

NANO="${NANO:-./bin/nano}"
EXAMPLE="examples/properties/sort_idempotent.nano"
PASS_ONLY_FILE="$(mktemp /tmp/proptest_pass_XXXXXX.nano)"

VERBOSE=0
for arg in "$@"; do
    [ "$arg" = "--verbose" ] && VERBOSE=1
done

log() { [ "$VERBOSE" -eq 1 ] && echo "$@" >&2 || true; }

fail() { echo "FAIL: $*" >&2; rm -f "$PASS_ONLY_FILE"; exit 1; }
pass() { echo "PASS: $*"; }

# Check binary exists
if [ ! -x "$NANO" ]; then
    fail "'$NANO' not found or not executable. Run 'make' first."
fi

# ── Test 1: passing properties only ──────────────────────────────────────
# Create a file with only valid properties
cat > "$PASS_ONLY_FILE" <<'NANO_EOF'
# Property file with only true properties
fn prop_add_commutative(a: int, b: int) -> bool {
    return (== (+ a b) (+ b a))
}

fn prop_mul_identity(a: int) -> bool {
    return (== (* a 1) a)
}

fn prop_bool_not_not(x: bool) -> bool {
    return (== (not (not x)) x)
}

fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
NANO_EOF

log "Test 1: All-passing properties should exit 0 and print PASS"
output=$("$NANO" --proptest --proptest-seed 42 "$PASS_ONLY_FILE" 2>&1) || {
    fail "Test 1: Expected exit 0 for all-passing properties, got exit $?"
}
if ! echo "$output" | grep -q "PASS"; then
    fail "Test 1: Expected 'PASS' in output, got: $output"
fi
if echo "$output" | grep -q "FAIL"; then
    fail "Test 1: Unexpected 'FAIL' in output: $output"
fi
pass "Test 1: All-passing properties → exit 0, output contains PASS"
log "$output"

# ── Test 2: known-broken property → exit 1 ───────────────────────────────
log "Test 2: Broken property should exit 1 and print FAIL"
BROKEN_FILE="$(mktemp /tmp/proptest_broken_XXXXXX.nano)"
cat > "$BROKEN_FILE" <<'NANO_EOF'
# Deliberately false: subtraction is NOT commutative
fn prop_broken_sub(a: int, b: int) -> bool {
    return (== (- a b) (- b a))
}

fn main() -> int { return 0 }
shadow main { assert (== (main) 0) }
NANO_EOF

broken_output=$("$NANO" --proptest --proptest-seed 42 "$BROKEN_FILE" 2>&1) && broken_exit=0 || broken_exit=$?
rm -f "$BROKEN_FILE"

if [ "$broken_exit" -eq 0 ]; then
    fail "Test 2: Expected exit 1 for broken property, got exit 0. Output: $broken_output"
fi
if ! echo "$broken_output" | grep -q "FAIL"; then
    fail "Test 2: Expected 'FAIL' in output, got: $broken_output"
fi
pass "Test 2: Broken property → exit 1, output contains FAIL"
log "$broken_output"

# ── Test 3: full example file (has broken property → exit 1) ─────────────
log "Test 3: Full example file contains a broken property"
full_output=$("$NANO" --proptest --proptest-seed 42 "$EXAMPLE" 2>&1) && full_exit=0 || full_exit=$?

if [ "$full_exit" -eq 0 ]; then
    fail "Test 3: Expected exit 1 (broken property in example), got exit 0. Output: $full_output"
fi
if ! echo "$full_output" | grep -q "PASS"; then
    fail "Test 3: Expected at least one 'PASS' in output, got: $full_output"
fi
if ! echo "$full_output" | grep -q "FAIL"; then
    fail "Test 3: Expected 'FAIL' for broken property in output, got: $full_output"
fi
pass "Test 3: Example file correctly reports mixed PASS/FAIL"
log "$full_output"

# ── Cleanup ───────────────────────────────────────────────────────────────
rm -f "$PASS_ONLY_FILE"

echo ""
echo "All proptest smoke tests passed."
