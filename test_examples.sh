#!/usr/bin/env bash
# Test runner for nanolang examples with timeout support
# Works on both Linux and macOS

set -u

TIMEOUT=5  # seconds
NANO="./bin/nano"
EXAMPLES_DIR="examples"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Counters
TOTAL=0
PASSED=0
FAILED=0
TIMEOUT_COUNT=0

echo "=========================================="
echo "  nanolang Example Test Runner"
echo "=========================================="
echo ""

# Function to run a single test with timeout
run_test() {
    local file="$1"
    local basename=$(basename "$file")
    
    TOTAL=$((TOTAL + 1))
    
    # Create temp files for output
    local stdout_file=$(mktemp)
    local stderr_file=$(mktemp)
    
    # Run the command in background
    $NANO "$file" --call main > "$stdout_file" 2> "$stderr_file" &
    local pid=$!
    
    # Wait for completion or timeout
    local elapsed=0
    while [ $elapsed -lt $TIMEOUT ]; do
        if ! kill -0 $pid 2>/dev/null; then
            # Process finished
            wait $pid
            local exit_code=$?
            
            if [ $exit_code -eq 0 ]; then
                echo -e "${GREEN}✅ PASS${NC} $basename"
                PASSED=$((PASSED + 1))
            else
                echo -e "${RED}❌ FAIL${NC} $basename (exit code: $exit_code)"
                if [ -s "$stderr_file" ]; then
                    echo "   Error output:"
                    head -3 "$stderr_file" | sed 's/^/   /'
                fi
                FAILED=$((FAILED + 1))
            fi
            
            rm -f "$stdout_file" "$stderr_file"
            return
        fi
        sleep 0.1
        elapsed=$((elapsed + 1))
    done
    
    # Timeout - kill the process
    kill -9 $pid 2>/dev/null
    wait $pid 2>/dev/null
    
    echo -e "${YELLOW}⏱️  TIMEOUT${NC} $basename (>${TIMEOUT}s)"
    if [ -s "$stdout_file" ]; then
        echo "   Last output:"
        tail -5 "$stdout_file" | sed 's/^/   /'
    fi
    TIMEOUT_COUNT=$((TIMEOUT_COUNT + 1))
    
    rm -f "$stdout_file" "$stderr_file"
}

# Run all tests
for file in $EXAMPLES_DIR/*.nano; do
    if [ -f "$file" ]; then
        run_test "$file"
    fi
done

# Print summary
echo ""
echo "=========================================="
echo "  Test Summary"
echo "=========================================="
echo -e "Total:    $TOTAL"
echo -e "${GREEN}Passed:   $PASSED${NC}"
echo -e "${RED}Failed:   $FAILED${NC}"
echo -e "${YELLOW}Timeout:  $TIMEOUT_COUNT${NC}"
echo ""

if [ $FAILED -eq 0 ] && [ $TIMEOUT_COUNT -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some tests need attention${NC}"
    exit 1
fi

