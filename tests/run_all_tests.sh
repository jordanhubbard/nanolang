#!/usr/bin/env bash
# Comprehensive test runner for nanolang
# Tests all language features with both interpreter and compiler

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

INTERPRETER_PASS=0
INTERPRETER_FAIL=0
COMPILER_PASS=0
COMPILER_FAIL=0

echo "========================================"
echo "NANOLANG COMPREHENSIVE TEST SUITE"
echo "========================================"
echo ""

# Run a single test with interpreter
run_interpreter_test() {
    local test_file="$1"
    local test_name=$(basename "$test_file")
    
    if ./bin/nano "$test_file" > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} INT: $test_name"
        ((INTERPRETER_PASS++))
        return 0
    else
        echo -e "${RED}❌${NC} INT: $test_name"
        ((INTERPRETER_FAIL++))
        return 1
    fi
}

# Run a single test with compiler
run_compiler_test() {
    local test_file="$1"
    local test_name=$(basename "$test_file" .nano)
    local output_file="/tmp/nanolang_test_$$_$test_name"
    
    if ./bin/nanoc "$test_file" -o "$output_file" > /dev/null 2>&1; then
        if [ -f "$output_file" ]; then
            if "$output_file" > /dev/null 2>&1; then
                echo -e "${GREEN}✅${NC} COM: $test_name"
                ((COMPILER_PASS++))
                rm -f "$output_file" "${output_file}.c"
                return 0
            fi
        fi
    fi
    
    echo -e "${RED}❌${NC} COM: $test_name"
    ((COMPILER_FAIL++))
    rm -f "$output_file" "${output_file}.c"
    return 1
}

echo "=== UNIT TESTS ==="
for test in tests/unit/*.nano; do
    [ -f "$test" ] || continue
    run_interpreter_test "$test"
    run_compiler_test "$test"
done

echo ""
echo "=== TUPLE TESTS ==="
for test in tests/tuple*.nano; do
    [ -f "$test" ] || continue
    run_interpreter_test "$test"
    run_compiler_test "$test"
done

echo ""
echo "=== EXAMPLES (as tests) ==="
for example in examples/*.nano; do
    [ -f "$example" ] || continue
    # Skip SDL examples (require external libraries)
    if [[ "$example" == *"sdl"* ]]; then
        continue
    fi
    run_interpreter_test "$example"
    run_compiler_test "$example"
done

echo ""
echo "========================================"
echo "TEST RESULTS SUMMARY"
echo "========================================"
echo ""
echo "Interpreter:"
echo -e "  ${GREEN}✅ Passed: $INTERPRETER_PASS${NC}"
if [ $INTERPRETER_FAIL -gt 0 ]; then
    echo -e "  ${RED}❌ Failed: $INTERPRETER_FAIL${NC}"
else
    echo "  ❌ Failed: 0"
fi

echo ""
echo "Compiler:"
echo -e "  ${GREEN}✅ Passed: $COMPILER_PASS${NC}"
if [ $COMPILER_FAIL -gt 0 ]; then
    echo -e "  ${RED}❌ Failed: $COMPILER_FAIL${NC}"
else
    echo "  ❌ Failed: 0"
fi

echo ""
TOTAL_PASS=$((INTERPRETER_PASS + COMPILER_PASS))
TOTAL_FAIL=$((INTERPRETER_FAIL + COMPILER_FAIL))
TOTAL=$((TOTAL_PASS + TOTAL_FAIL))
echo -e "TOTAL: ${GREEN}$TOTAL_PASS passed${NC}, ${RED}$TOTAL_FAIL failed${NC} out of $TOTAL tests"

if [ $TOTAL_FAIL -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 ALL TESTS PASSED!${NC}"
    exit 0
else
    echo ""
    echo -e "${YELLOW}⚠️  Some tests failed${NC}"
    exit 1
fi

