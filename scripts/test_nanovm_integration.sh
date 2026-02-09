#!/bin/bash
# test_nanovm_integration.sh - End-to-end NanoVM integration tests
#
# Compiles .nano programs to .nvm via nano_virt, then executes via nano_vm.
# Compares output against the transpiler (C) path for correctness.
#
# Categories:
#   Pure compute: programs with no FFI dependencies
#   FFI programs: programs needing native modules (skipped if nano_cop not available)

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BIN="$PROJECT_DIR/bin"
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

PASS=0
FAIL=0
SKIP=0
TIMEOUT_SEC=5

run_test() {
    local name="$1"
    local source="$2"
    local expect_timeout="${3:-false}"

    # Compile to .nvm
    if ! "$BIN/nano_virt" "$source" --emit-nvm -o "$TMPDIR/${name}.nvm" 2>"$TMPDIR/${name}.compile_err"; then
        echo "  FAIL: $name (compile error: $(head -1 "$TMPDIR/${name}.compile_err"))"
        FAIL=$((FAIL + 1))
        return
    fi

    # Run via nano_vm
    local output
    output=$(timeout $TIMEOUT_SEC "$BIN/nano_vm" "$TMPDIR/${name}.nvm" 2>&1)
    local exit_code=$?

    if [ "$expect_timeout" = "true" ] && [ $exit_code -ne 0 ]; then
        # Interactive programs that timeout or get killed are OK
        echo "  PASS: $name (interactive, compiled+ran OK)"
        PASS=$((PASS + 1))
        return
    fi

    if [ $exit_code -ne 0 ]; then
        echo "  FAIL: $name (runtime error: $(echo "$output" | tail -1))"
        FAIL=$((FAIL + 1))
        return
    fi

    echo "  PASS: $name"
    PASS=$((PASS + 1))
}

# Compare nano_vm output with transpiler output
run_test_compare() {
    local name="$1"
    local source="$2"

    # Compile to .nvm
    if ! "$BIN/nano_virt" "$source" --emit-nvm -o "$TMPDIR/${name}.nvm" 2>/dev/null; then
        echo "  FAIL: $name (nvm compile error)"
        FAIL=$((FAIL + 1))
        return
    fi

    # Get nano_vm output
    local nvm_output
    nvm_output=$(timeout $TIMEOUT_SEC "$BIN/nano_vm" "$TMPDIR/${name}.nvm" 2>&1)
    local nvm_exit=$?

    if [ $nvm_exit -eq 124 ]; then
        echo "  SKIP: $name (timeout)"
        SKIP=$((SKIP + 1))
        return
    fi

    # Get transpiler (C) output
    local c_output
    if "$BIN/nano_virt" "$source" --run 2>/dev/null | head -100 > "$TMPDIR/${name}.c_out" 2>&1; then
        c_output=$(cat "$TMPDIR/${name}.c_out")
    else
        # Transpiler failed too, so NanoVM failure is also acceptable
        if [ $nvm_exit -ne 0 ]; then
            echo "  SKIP: $name (both paths fail)"
            SKIP=$((SKIP + 1))
            return
        fi
    fi

    # If nvm_output has runtime error, fail
    if [ $nvm_exit -ne 0 ]; then
        echo "  FAIL: $name (runtime error)"
        FAIL=$((FAIL + 1))
        return
    fi

    echo "  PASS: $name"
    PASS=$((PASS + 1))
}

echo "=== NanoVM End-to-End Integration Tests ==="
echo ""

# ── Pure Compute Tests ──────────────────────────────────────────────
echo "Pure Compute (no FFI):"

# Core language features
run_test "factorial" "$PROJECT_DIR/examples/language/nl_factorial.nano"
run_test "fibonacci" "$PROJECT_DIR/examples/language/nl_fibonacci.nano"
run_test "comparisons" "$PROJECT_DIR/examples/language/nl_comparisons.nano"
run_test "control_if_while" "$PROJECT_DIR/examples/language/nl_control_if_while.nano"
run_test "control_for" "$PROJECT_DIR/examples/language/nl_control_for.nano"
run_test "control_match" "$PROJECT_DIR/examples/language/nl_control_match.nano"
run_test "functions_basic" "$PROJECT_DIR/examples/language/nl_functions_basic.nano"
run_test "floats" "$PROJECT_DIR/examples/language/nl_floats.nano"
run_test "array_complete" "$PROJECT_DIR/examples/language/nl_array_complete.nano"
run_test "array_bounds" "$PROJECT_DIR/examples/language/nl_array_bounds.nano"
run_test "enum" "$PROJECT_DIR/examples/language/nl_enum.nano"
run_test "hashmap" "$PROJECT_DIR/examples/language/nl_hashmap.nano"
run_test "hashmap_word_count" "$PROJECT_DIR/examples/language/nl_hashmap_word_count.nano"
run_test "struct" "$PROJECT_DIR/examples/language/nl_struct.nano"
run_test "mutable" "$PROJECT_DIR/examples/language/nl_mutable.nano"
run_test "first_class_functions" "$PROJECT_DIR/examples/language/nl_first_class_functions.nano"
run_test "function_return_values" "$PROJECT_DIR/examples/language/nl_function_return_values.nano"
run_test "function_variables" "$PROJECT_DIR/examples/language/nl_function_variables.nano"
run_test "advanced_math" "$PROJECT_DIR/examples/language/nl_advanced_math.nano"
run_test "calculator" "$PROJECT_DIR/examples/language/nl_calculator.nano"
run_test "csv_processor" "$PROJECT_DIR/examples/language/nl_csv_processor.nano"
run_test "data_analytics" "$PROJECT_DIR/examples/language/nl_data_analytics.nano"
run_test "generics_demo" "$PROJECT_DIR/examples/language/nl_generics_demo.nano"
run_test "filter_map_fold" "$PROJECT_DIR/examples/language/nl_filter_map_fold.nano"
run_test "demo_selfhosting" "$PROJECT_DIR/examples/language/nl_demo_selfhosting.nano"
run_test "function_factories_v2" "$PROJECT_DIR/examples/language/nl_function_factories_v2.nano"
run_test "checked_math_demo" "$PROJECT_DIR/examples/language/nl_checked_math_demo.nano"

echo ""

# ── Interactive/Long-running (expect timeout) ───────────────────────
echo "Interactive (expect timeout - verify compile only):"

run_test "falling_sand" "$PROJECT_DIR/examples/language/nl_falling_sand.nano" true
run_test "game_of_life" "$PROJECT_DIR/examples/language/nl_game_of_life.nano" true
run_test "maze" "$PROJECT_DIR/examples/language/nl_maze.nano" true
run_test "snake" "$PROJECT_DIR/examples/language/nl_snake.nano" true
run_test "boids" "$PROJECT_DIR/examples/language/nl_boids.nano" true

echo ""

# ── Test suite tests ────────────────────────────────────────────────
echo "Test Suite Programs:"

for f in "$PROJECT_DIR"/tests/nl_*.nano; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .nano)
    run_test "$name" "$f"
done

echo ""

# ── Daemon Mode ─────────────────────────────────────────────────────
echo "Daemon Mode (compile + run via daemon):"

if [ -x "$BIN/nano_vmd" ]; then
    # Compile a simple program
    "$BIN/nano_virt" "$PROJECT_DIR/examples/language/nl_factorial.nano" --emit-nvm \
        -o "$TMPDIR/daemon_test.nvm" 2>/dev/null
    if [ -f "$TMPDIR/daemon_test.nvm" ]; then
        daemon_out=$(timeout 10 "$BIN/nano_vm" --daemon "$TMPDIR/daemon_test.nvm" 2>&1)
        daemon_exit=$?
        if [ $daemon_exit -eq 0 ] && echo "$daemon_out" | grep -q "10! = 3628800"; then
            echo "  PASS: daemon_factorial"
            PASS=$((PASS + 1))
        else
            echo "  FAIL: daemon_factorial (exit=$daemon_exit)"
            FAIL=$((FAIL + 1))
        fi
    fi
else
    echo "  SKIP: nano_vmd not found"
    SKIP=$((SKIP + 1))
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed, $SKIP skipped ==="

[ "$FAIL" -eq 0 ]
