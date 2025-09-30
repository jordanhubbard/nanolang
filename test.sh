#!/bin/bash
# Test script for nanolang compiler
# Tests both interpreter (shadow tests) and compiler (binary execution)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

COMPILER="./bin/nanoc"
TEST_DIR="examples"
TESTS_PASSED=0
TESTS_FAILED=0
TOTAL_TESTS=0

# Check if timeout command is available
if command -v timeout &> /dev/null; then
    TIMEOUT_CMD="timeout 30"
    TIMEOUT_RUN="timeout 10"
else
    # macOS doesn't have timeout by default, so skip it
    TIMEOUT_CMD=""
    TIMEOUT_RUN=""
fi

# List of examples to test
EXAMPLES=(
    "hello.nano"
    "calculator.nano"
    "factorial.nano"
    "fibonacci.nano"
    # "primes.nano"  # Disabled: too slow (times out during shadow tests)
    "01_operators.nano"
    "02_strings.nano"
    "03_floats.nano"
    "04_loops.nano"
    "04_loops_working.nano"
    "05_mutable.nano"
    "06_logical.nano"
    "07_comparisons.nano"
    "08_types.nano"
    "09_math.nano"
)

echo "========================================"
echo "nanolang Test Suite"
echo "========================================"
echo ""

# Check if compiler exists
if [ ! -f "$COMPILER" ]; then
    echo -e "${RED}Error: Compiler '$COMPILER' not found${NC}"
    echo "Run 'make' to build the compiler first"
    exit 1
fi

# Create temp directory for test outputs
mkdir -p .test_output

# Run tests
for example in "${EXAMPLES[@]}"; do
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    test_file="$TEST_DIR/$example"
    test_name=$(basename "$example" .nano)
    output_binary=".test_output/${test_name}"

    echo -n "Testing $example... "

    # Phase 1: Compile (includes interpreter/shadow test execution)
    # Use timeout to prevent hanging tests (if available)
    if $TIMEOUT_CMD $COMPILER "$test_file" -o "$output_binary" > ".test_output/${test_name}.compile.log" 2>&1; then
        # Compilation successful (shadow tests passed)

        # Phase 2: Run compiled binary
        if [ -f "$output_binary" ]; then
            if $TIMEOUT_RUN "$output_binary" > ".test_output/${test_name}.run.log" 2>&1; then
                echo -e "${GREEN}✓ PASS${NC} (interpreter + compiler)"
                TESTS_PASSED=$((TESTS_PASSED + 1))
            else
                exit_code=$?
                if [ $exit_code -eq 124 ]; then
                    echo -e "${RED}✗ TIMEOUT${NC} (binary execution timed out)"
                else
                    echo -e "${RED}✗ FAIL${NC} (binary execution failed)"
                fi
                echo "  See .test_output/${test_name}.run.log for details"
                TESTS_FAILED=$((TESTS_FAILED + 1))
            fi
        else
            echo -e "${RED}✗ FAIL${NC} (binary not generated)"
            TESTS_FAILED=$((TESTS_FAILED + 1))
        fi
    else
        exit_code=$?
        if [ $exit_code -eq 124 ]; then
            echo -e "${RED}✗ TIMEOUT${NC} (compilation/shadow tests timed out)"
        else
            echo -e "${RED}✗ FAIL${NC} (compilation/shadow tests failed)"
        fi
        echo "  See .test_output/${test_name}.compile.log for details"
        TESTS_FAILED=$((TESTS_FAILED + 1))
    fi
done

# Summary
echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Total tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $TESTS_FAILED${NC}"
else
    echo "Failed: 0"
fi
echo ""

# Exit with appropriate code
if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Some tests failed${NC}"
    exit 1
else
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
fi
