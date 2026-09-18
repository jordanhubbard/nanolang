#!/bin/sh
# Run all self-hosted compiler tests

set -e

TESTS_DIR="tests/selfhost"
NANOC="${NANOLANG_SELFHOST_COMPILER:-${NANOC:-./bin/nanoc_stage1}}"
LOG_DIR=".test_output/selfhost"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "SELF-HOSTED COMPILER TEST SUITE"
echo "Compiler: $NANOC"
echo "========================================"
echo ""

# Check if compiler exists
if [ ! -f "$NANOC" ]; then
    echo "❌ Error: Compiler not found at $NANOC"
    exit 1
fi

# Positive tests (should compile and run)
TESTS="
test_arithmetic_ops.nano
test_comparison_ops.nano
test_logical_ops.nano
test_while_loops.nano
test_loop_control.nano
test_recursion.nano
test_function_calls.nano
test_returned_function_calls.nano
test_opaque_null_argument.nano
test_nested_array_indexing.nano
test_let_set.nano
test_if_else.nano
test_match_bindings.nano
test_match_expression_blocks.nano
test_infix_ops.nano
"

# NOTE: test_std_modules_env_fs_binary.nano disabled due to Ubuntu linking issue
# Works on macOS but fails on Ubuntu with undefined references to std_env__* functions
# Regular test suite already covers this functionality (tests/test_std_modules_env_fs_binary.nano works)
# TODO: Investigate module linking differences between platforms in selfhost tests

# Negative tests (compiler should reject)
NEGATIVE_TESTS="
test_requires_bool.nano
test_function_arg_type_errors.nano
test_returned_function_arg_type_error.nano
test_returned_function_arity_error.nano
test_opaque_nonzero_argument.nano
"

PASSED=0
FAILED=0

for test in $TESTS; do
    TEST_PATH="$TESTS_DIR/$test"
    TEST_BIN="bin/selfhost_$(basename $test .nano)"
    
    printf "Testing %-30s ... " "$test"
    
    # Compile (timeout to avoid nanoc infinite loops)
    COMPILE_LOG="$LOG_DIR/$(basename "$test" .nano).compile.log"
    if perl -e 'alarm 60; exec @ARGV' "$NANOC" "$TEST_PATH" -o "$TEST_BIN" > "$COMPILE_LOG" 2>&1; then
        # Run
        if [ "$test" = "test_returned_function_calls.nano" ]; then
            OUTPUT=$(perl -e 'alarm 60; exec @ARGV' $TEST_BIN 2>&1) || OUTPUT_STATUS=$?
            OUTPUT_STATUS=${OUTPUT_STATUS:-0}
            if [ "$OUTPUT_STATUS" -eq 0 ] && [ "$OUTPUT" = "callee
argument" ]; then
                echo "✅ PASS"
                PASSED=$((PASSED + 1))
            else
                echo "❌ FAIL (runtime order/error)"
                FAILED=$((FAILED + 1))
            fi
            unset OUTPUT_STATUS
        elif perl -e 'alarm 60; exec @ARGV' $TEST_BIN > /dev/null 2>&1; then
            echo "✅ PASS"
            PASSED=$((PASSED + 1))
        else
            echo "❌ FAIL (runtime error)"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "❌ FAIL (compilation error)"
        cat "$COMPILE_LOG"
        FAILED=$((FAILED + 1))
    fi
done

for test in $NEGATIVE_TESTS; do
    TEST_PATH="$TESTS_DIR/$test"
    TEST_BIN="bin/selfhost_$(basename $test .nano)"

    printf "Testing %-30s ... " "$test"

    case "$test" in
        test_requires_bool.nano)
            EXPECTED_DIAGNOSTIC="[E0001] assert condition must be bool"
            ;;
        test_function_arg_type_errors.nano)
            EXPECTED_DIAGNOSTIC="[E0010] Argument 1 to 'add': expected int, got string"
            ;;
        test_returned_function_arg_type_error.nano)
            EXPECTED_DIAGNOSTIC="[E0010] Argument 1 to the function expression: expected int, got string"
            ;;
        test_returned_function_arity_error.nano)
            EXPECTED_DIAGNOSTIC="[E0010] The function expression expects 1 argument(s), but I see 2."
            ;;
        test_opaque_nonzero_argument.nano)
            EXPECTED_DIAGNOSTIC="[E0010] Argument 1 to 'is_null': expected SDL_Window, got int"
            ;;
        *)
            echo "❌ FAIL (missing rejection contract)"
            FAILED=$((FAILED + 1))
            continue
            ;;
    esac

    COMPILE_LOG="$LOG_DIR/$(basename "$test" .nano).compile.log"
    rm -f "$TEST_BIN"
    if python3 "$TESTS_DIR/expect_rejection.py" \
        --timeout 60 --log "$COMPILE_LOG" --output "$TEST_BIN" \
        --require "$EXPECTED_DIAGNOSTIC" -- \
        "$NANOC" "$TEST_PATH" -o "$TEST_BIN"; then
        echo "✅ EXPECTED FAIL"
        PASSED=$((PASSED + 1))
    else
        echo "❌ FAIL (not the required semantic rejection)"
        cat "$COMPILE_LOG"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "========================================"
if NANOLANG_SELFHOST_COMPILER="$NANOC" python3 tests/test_selfhost_import_paths.py; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi
if NANOLANG_SELFHOST_COMPILER="$NANOC" python3 tests/test_selfhost_cli.py; then
    PASSED=$((PASSED + 1))
else
    FAILED=$((FAILED + 1))
fi
echo "Results: $PASSED passed, $FAILED failed (including import-path and CLI suites)"
echo "========================================"

# Cleanup intermediate test binaries
echo ""
echo "Cleaning up test binaries..."
for test in $TESTS $NEGATIVE_TESTS; do
    TEST_BIN="bin/selfhost_$(basename $test .nano)"
    if [ -f "$TEST_BIN" ]; then
        rm -f "$TEST_BIN"
    fi
done
echo "✓ Removed selfhost_test_* binaries"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo "🎉 All tests passed!"
    exit 0
else
    echo ""
    echo "❌ Some tests failed"
    exit 1
fi
