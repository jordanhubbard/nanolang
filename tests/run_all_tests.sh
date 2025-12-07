#!/usr/bin/env bash
# Comprehensive test runner for nanolang
# Tests all language features with both interpreter and compiler

# Don't exit on first error - show all test results
set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

INTERPRETER_PASS=0
INTERPRETER_FAIL=0
COMPILER_PASS=0
COMPILER_FAIL=0
INTERPRETER_SKIP=0
COMPILER_SKIP=0

# Expected failures: Tests for unimplemented features
# Format: "filename:mode" where mode is "int", "com", or "both"
EXPECTED_FAILURES=(
    "test_firstclass_functions.nano:both"   # First-class functions not implemented
    "test_unions_match_comprehensive.nano:com"  # Union transpilation ordering issue
)

echo "========================================"
echo "NANOLANG COMPREHENSIVE TEST SUITE"
echo "========================================"
echo ""

# Check if test should be skipped
should_skip_test() {
    local test_file="$1"
    local mode="$2"  # "int" or "com"
    local test_name=$(basename "$test_file")
    
    for entry in "${EXPECTED_FAILURES[@]}"; do
        local file_part="${entry%%:*}"
        local mode_part="${entry##*:}"
        
        if [ "$test_name" = "$file_part" ]; then
            if [ "$mode_part" = "both" ] || [ "$mode_part" = "$mode" ]; then
                return 0  # Should skip
            fi
        fi
    done
    
    return 1  # Should not skip
}

# Run a single test with interpreter
run_interpreter_test() {
    local test_file="$1"
    local test_name=$(basename "$test_file")
    
    # Check if this test should be skipped
    if should_skip_test "$test_file" "int"; then
        echo -e "${BLUE}⊘${NC} INT: $test_name ${YELLOW}(expected failure - feature not implemented)${NC}"
        INTERPRETER_SKIP=$((INTERPRETER_SKIP + 1))
        return 0
    fi
    
    if ./bin/nano "$test_file" > /dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} INT: $test_name"
        INTERPRETER_PASS=$((INTERPRETER_PASS + 1))
        return 0
    else
        echo -e "${RED}❌${NC} INT: $test_name"
        INTERPRETER_FAIL=$((INTERPRETER_FAIL + 1))
        return 1
    fi
}

# Run a single test with compiler
run_compiler_test() {
    local test_file="$1"
    local test_name=$(basename "$test_file" .nano)
    local output_file="/tmp/nanolang_test_$$_$test_name"
    
    # Check if this test should be skipped
    if should_skip_test "$test_file" "com"; then
        echo -e "${BLUE}⊘${NC} COM: $test_name ${YELLOW}(expected failure - feature not implemented)${NC}"
        COMPILER_SKIP=$((COMPILER_SKIP + 1))
        return 0
    fi
    
    if ./bin/nanoc "$test_file" -o "$output_file" > /dev/null 2>&1; then
        if [ -f "$output_file" ]; then
            if "$output_file" > /dev/null 2>&1; then
                echo -e "${GREEN}✅${NC} COM: $test_name"
                COMPILER_PASS=$((COMPILER_PASS + 1))
                rm -f "$output_file" "${output_file}.c"
                return 0
            fi
        fi
    fi
    
    echo -e "${RED}❌${NC} COM: $test_name"
    COMPILER_FAIL=$((COMPILER_FAIL + 1))
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
echo "(Skipped - run 'make examples' to test separately)"
# for example in examples/*.nano; do
#     [ -f "$example" ] || continue
#     # Skip SDL examples (require external libraries)
#     if [[ "$example" == *"sdl"* ]]; then
#         continue
#     fi
#     run_interpreter_test "$example"
#     run_compiler_test "$example"
# done

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
if [ $INTERPRETER_SKIP -gt 0 ]; then
    echo -e "  ${BLUE}⊘  Skipped: $INTERPRETER_SKIP${NC} ${YELLOW}(expected failures)${NC}"
fi

echo ""
echo "Compiler:"
echo -e "  ${GREEN}✅ Passed: $COMPILER_PASS${NC}"
if [ $COMPILER_FAIL -gt 0 ]; then
    echo -e "  ${RED}❌ Failed: $COMPILER_FAIL${NC}"
else
    echo "  ❌ Failed: 0"
fi
if [ $COMPILER_SKIP -gt 0 ]; then
    echo -e "  ${BLUE}⊘  Skipped: $COMPILER_SKIP${NC} ${YELLOW}(expected failures)${NC}"
fi

echo ""
TOTAL_PASS=$((INTERPRETER_PASS + COMPILER_PASS))
TOTAL_FAIL=$((INTERPRETER_FAIL + COMPILER_FAIL))
TOTAL_SKIP=$((INTERPRETER_SKIP + COMPILER_SKIP))
TOTAL=$((TOTAL_PASS + TOTAL_FAIL + TOTAL_SKIP))
echo -e "TOTAL: ${GREEN}$TOTAL_PASS passed${NC}, ${RED}$TOTAL_FAIL failed${NC}, ${BLUE}$TOTAL_SKIP skipped${NC} out of $TOTAL tests"

if [ $TOTAL_FAIL -eq 0 ]; then
    echo ""
    if [ $TOTAL_SKIP -gt 0 ]; then
        echo -e "${GREEN}✅ All runnable tests passed!${NC} ${YELLOW}($TOTAL_SKIP skipped due to unimplemented features)${NC}"
    else
        echo -e "${GREEN}🎉 ALL TESTS PASSED!${NC}"
    fi
    exit 0
else
    echo ""
    echo -e "${YELLOW}⚠️  Some tests failed${NC}"
    exit 1
fi

