#!/usr/bin/env bash
# Union Types Test Runner
# Tests parser and type checker for union types

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
NANOC="$REPO_ROOT/bin/nanoc"
NANO="$REPO_ROOT/bin/nano"
TEST_DIR="$SCRIPT_DIR"

if [ ! -f "$NANOC" ]; then
    echo "Error: nanoc not found at $NANOC"
    echo "Please run 'make' first"
    exit 1
fi

echo "========================================="
echo "Union Types Test Suite"
echo "========================================="
echo ""

PASSED=0
FAILED=0
TOTAL=0

run_test() {
    local test_file=$1
    local test_name=$(basename "$test_file" .nano)
    
    TOTAL=$((TOTAL + 1))
    
    echo -n "Test $TOTAL: $test_name ... "
    
    # Compile the test (suppress parser errors from interpreter)
    if $NANOC "$test_file" -o "/tmp/$test_name" 2>&1 | grep -q "Type checking failed"; then
        echo "❌ FAILED (type checking)"
        FAILED=$((FAILED + 1))
        return 1
    fi
    
    # Run the compiled binary
    if "/tmp/$test_name" > /dev/null 2>&1; then
        echo "✅ PASSED"
        PASSED=$((PASSED + 1))
        rm -f "/tmp/$test_name"
        return 0
    else
        echo "❌ FAILED (runtime)"
        FAILED=$((FAILED + 1))
        return 1
    fi
}

# Run all tests in order
for test_file in $(ls -1 $TEST_DIR/*.nano 2>/dev/null | sort); do
    run_test "$test_file"
done

echo ""
echo "========================================="
echo "Results: $PASSED passed, $FAILED failed, $TOTAL total"
echo "========================================="

if [ $FAILED -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Some tests failed"
    exit 1
fi

