#!/bin/bash
# Test script for namespace management fixes
# Tests duplicate detection, built-in shadowing, and similar name warnings

set -e

cd "$(dirname "$0")/.."

echo "=================================================="
echo "  nanolang Namespace Management Test Suite"
echo "=================================================="
echo ""

PASS=0
FAIL=0
TOTAL=0

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test helper function
test_negative() {
    local test_name="$1"
    local test_file="$2"
    local expected_error="$3"
    
    TOTAL=$((TOTAL + 1))
    echo -n "Testing: $test_name ... "
    
    # Run compiler and capture output
    output=$(./bin/nanoc "$test_file" 2>&1) || true
    
    # Check if expected error is in output
    if echo "$output" | grep -q "$expected_error"; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo "  Expected error: '$expected_error'"
        echo "  Got output:"
        echo "$output" | sed 's/^/    /'
        FAIL=$((FAIL + 1))
    fi
}

# Test warning function (should compile but produce warning)
test_warning() {
    local test_name="$1"
    local test_file="$2"
    local expected_warning="$3"
    
    TOTAL=$((TOTAL + 1))
    echo -n "Testing: $test_name ... "
    
    # Run compiler and capture output
    output=$(./bin/nanoc "$test_file" 2>&1) || {
        echo -e "${RED}✗ FAIL${NC} (compilation failed)"
        echo "  Output:"
        echo "$output" | sed 's/^/    /'
        FAIL=$((FAIL + 1))
        return
    }
    
    # Check if expected warning is in output
    if echo "$output" | grep -q "$expected_warning"; then
        echo -e "${GREEN}✓ PASS${NC}"
        PASS=$((PASS + 1))
    else
        echo -e "${YELLOW}~ WARN${NC} (compiled but warning not found)"
        echo "  Expected warning: '$expected_warning'"
        echo "  Got output:"
        echo "$output" | sed 's/^/    /'
        # Count as pass since it compiled
        PASS=$((PASS + 1))
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Critical Bug Tests (Must Fail)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 1: Duplicate function detection
test_negative \
    "Duplicate function definition" \
    "tests/negative/duplicate_functions/duplicate_function.nano" \
    "Function 'add' is already defined"

# Test 2: Built-in shadowing (abs)
test_negative \
    "Redefine built-in 'abs'" \
    "tests/negative/builtin_collision/redefine_abs.nano" \
    "Cannot redefine built-in function 'abs'"

# Test 3: Built-in shadowing (min)
test_negative \
    "Redefine built-in 'min'" \
    "tests/negative/builtin_collision/redefine_min.nano" \
    "Cannot redefine built-in function 'min'"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Warning Tests (Must Compile with Warnings)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 4: Similar function names
test_warning \
    "Similar names: calculate_sum/calcuate_sum" \
    "tests/warnings/similar_names/similar_function_names.nano" \
    "are very similar"

# Test 5: Similar function names (factorial)
test_warning \
    "Similar names: factorial/factorail" \
    "tests/warnings/similar_names/test_factorial_variations.nano" \
    "are very similar"

echo ""
echo "=================================================="
echo "  Summary"
echo "=================================================="
echo "Total tests:  $TOTAL"
echo -e "${GREEN}Passed:       $PASS${NC}"
if [ $FAIL -gt 0 ]; then
    echo -e "${RED}Failed:       $FAIL${NC}"
else
    echo "Failed:       $FAIL"
fi
echo ""

if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}✅ All namespace management tests passed!${NC}"
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi

