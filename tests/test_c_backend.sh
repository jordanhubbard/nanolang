#!/usr/bin/env bash
# tests/test_c_backend.sh — test the nanolang --target c backend
#
# For each test .nano file: compile with --target c, then compile the
# resulting .c with gcc, run it, and verify the output.
#
# Usage: ./tests/test_c_backend.sh [path/to/nanoc]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Allow passing the compiler path as first argument
NANOC="${1:-$REPO_ROOT/bin/nanoc_c}"

if [ ! -x "$NANOC" ]; then
    echo "ERROR: compiler not found at $NANOC"
    echo "Run 'make' first, or pass the compiler path as an argument."
    exit 1
fi

TMPDIR_TESTS="${TMPDIR:-/tmp}/nano_c_backend_tests_$$"
mkdir -p "$TMPDIR_TESTS"
trap 'rm -rf "$TMPDIR_TESTS"' EXIT

PASS=0
FAIL=0

run_test() {
    local name="$1"
    local nano_file="$2"
    local expected="$3"

    local base
    base="$(basename "$nano_file" .nano)"
    local c_out="$TMPDIR_TESTS/${base}.c"
    local exe_out="$TMPDIR_TESTS/${base}"

    printf "  %-30s " "$name"

    # Step 1: compile .nano → .c
    if ! "$NANOC" "$nano_file" --target c -o "$c_out" 2>/dev/null; then
        echo "FAIL (nanoc --target c failed)"
        FAIL=$((FAIL + 1))
        return
    fi

    # Step 2: compile .c → executable
    if ! gcc -std=gnu11 -o "$exe_out" "$c_out" 2>/dev/null; then
        echo "FAIL (gcc compilation failed)"
        FAIL=$((FAIL + 1))
        return
    fi

    # Step 3: run and capture output
    local actual
    actual="$("$exe_out" 2>/dev/null)" || true

    # Step 4: compare
    if [ "$actual" = "$expected" ]; then
        echo "PASS"
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        echo "    expected: $(printf '%q' "$expected")"
        echo "    actual:   $(printf '%q' "$actual")"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== nanolang --target c backend tests ==="
echo ""

TESTS_DIR="$SCRIPT_DIR/c_backend"

run_test "hello world"       "$TESTS_DIR/test01_hello.nano"      "hello world"
run_test "arithmetic (3+4)"  "$TESTS_DIR/test02_arithmetic.nano" "7"
run_test "if/else (max)"     "$TESTS_DIR/test03_ifelse.nano"     "20"
run_test "while (sum 1..5)"  "$TESTS_DIR/test04_while.nano"      "15"
run_test "string concat"     "$TESTS_DIR/test05_strings.nano"    "Hello, nano"

echo ""
echo "Results: $PASS passed, $FAIL failed"
echo ""

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
exit 0
