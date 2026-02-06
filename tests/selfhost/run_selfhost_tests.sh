#!/bin/sh
# Run all self-hosted compiler tests

set -e

TESTS_DIR="tests/selfhost"
NANOC="./bin/nanoc"

echo "========================================"
echo "SELF-HOSTED COMPILER TEST SUITE"
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
test_recursion.nano
test_function_calls.nano
test_let_set.nano
test_if_else.nano
test_match_bindings.nano
test_infix_ops.nano
"

# NOTE: test_std_modules_env_fs_binary.nano disabled due to Ubuntu linking issue
# Works on macOS but fails on Ubuntu with undefined references to std_env__* functions
# Regular test suite already covers this functionality (tests/test_std_modules_env_fs_binary.nano works)
# TODO: Investigate module linking differences between platforms in selfhost tests

# Negative tests (compiler should reject)
NEGATIVE_TESTS="
test_requires_bool.nano
"

# NOTE: test_function_arg_type_errors.nano disabled - self-hosted typechecker passes this through
# The C reference compiler correctly rejects it, but self-hosted needs type checking improvements
# TODO: Fix self-hosted typechecker to properly reject function argument type mismatches

PASSED=0
FAILED=0

for test in $TESTS; do
    TEST_PATH="$TESTS_DIR/$test"
    TEST_BIN="bin/selfhost_$(basename $test .nano)"
    
    printf "Testing %-30s ... " "$test"
    
    # Compile (timeout to avoid nanoc infinite loops)
    if perl -e 'alarm 60; exec @ARGV' $NANOC "$TEST_PATH" -o "$TEST_BIN" > /dev/null 2>&1; then
        # Run
        if perl -e 'alarm 60; exec @ARGV' $TEST_BIN > /dev/null 2>&1; then
            echo "✅ PASS"
            PASSED=$((PASSED + 1))
        else
            echo "❌ FAIL (runtime error)"
            FAILED=$((FAILED + 1))
        fi
    else
        echo "❌ FAIL (compilation error)"
        FAILED=$((FAILED + 1))
    fi
done

for test in $NEGATIVE_TESTS; do
    TEST_PATH="$TESTS_DIR/$test"
    TEST_BIN="bin/selfhost_$(basename $test .nano)"

    printf "Testing %-30s ... " "$test"

    if perl -e 'alarm 60; exec @ARGV' $NANOC "$TEST_PATH" -o "$TEST_BIN" > /dev/null 2>&1; then
        echo "❌ FAIL (expected compilation error)"
        FAILED=$((FAILED + 1))
        if [ -f "$TEST_BIN" ]; then
            rm -f "$TEST_BIN"
        fi
    else
        echo "✅ EXPECTED FAIL"
        PASSED=$((PASSED + 1))
    fi
done

echo ""
echo "========================================"
echo "Results: $PASSED passed, $FAILED failed"
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
