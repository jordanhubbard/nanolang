#!/bin/bash
# Negative Test Runner
# Tests that verify the compiler correctly REJECTS invalid code
set -e

NANOC="../../bin/nanoc"
PASSED=0
FAILED=0
TOTAL=0

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================================"
echo "Running Negative Tests (Should Fail to Compile)"
echo "================================================"
echo ""

# Find all .nano files in negative test directories
for file in $(find . -name "*.nano" -type f | sort); do
    TOTAL=$((TOTAL + 1))
    filename=$(basename "$file")
    category=$(dirname "$file" | sed 's|^\./||')
    
    printf "%-50s " "$category/$filename"
    
    # Try to compile - it SHOULD fail
    if $NANOC "$file" -o /tmp/negative_test_$$  2>/dev/null; then
        # Compilation succeeded - that's bad!
        printf "${RED}FAIL${NC} (compiled when it shouldn't)\n"
        FAILED=$((FAILED + 1))
        rm -f /tmp/negative_test_$$
    else
        # Compilation failed - that's good!
        printf "${GREEN}PASS${NC} (rejected as expected)\n"
        PASSED=$((PASSED + 1))
    fi
done

echo ""
echo "================================================"
echo "Negative Test Results:"
echo "  Total:  $TOTAL"
echo "  Passed: $PASSED (compiler correctly rejected invalid code)"
echo "  Failed: $FAILED (compiler accepted invalid code!)"
echo "================================================"

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All negative tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some negative tests failed!${NC}"
    exit 1
fi
