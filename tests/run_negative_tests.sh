#!/bin/bash
# Test runner for negative tests
# These tests should FAIL to compile

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

COMPILER="../bin/nanoc"
NEGATIVE_DIR="negative"
PASSED=0
FAILED=0
TOTAL=0

echo "========================================" 
echo "Running Negative Tests"
echo "========================================"
echo ""

# Check if compiler exists
if [ ! -f "$COMPILER" ]; then
    echo -e "${RED}Error: Compiler '$COMPILER' not found${NC}"
    echo "Run 'make' from the project root first"
    exit 1
fi

# Find all negative test files
find "$NEGATIVE_DIR" -name "*.nano" | sort | while read -r test_file; do
    TOTAL=$((TOTAL + 1))
    test_name=$(basename "$test_file" .nano)
    category=$(basename $(dirname "$test_file"))
    
    echo -n "Testing $category/$test_name... "
    
    # Try to compile - should fail
    if $COMPILER "$test_file" -o /tmp/negative_test_output 2>/dev/null >/dev/null; then
        echo -e "${RED}✗ FAIL${NC} (compiled when it should have failed)"
        FAILED=$((FAILED + 1))
    else
        echo -e "${GREEN}✓ PASS${NC} (failed as expected)"
        PASSED=$((PASSED + 1))
    fi
    
    # Clean up
    rm -f /tmp/negative_test_output
done

echo ""
echo "========================================"
echo "Negative Test Summary"
echo "========================================"
echo "Total: $TOTAL"
echo -e "${GREEN}Passed: $PASSED${NC}"

if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
    exit 1
else
    echo "Failed: 0"
    echo -e "${GREEN}All negative tests passed!${NC}"
    exit 0
fi

