#!/usr/bin/env bash
#
# Nanolang Performance Benchmarking Suite
#
# Measures compilation speed, runtime performance, and memory usage
# to track performance over time and catch regressions.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_ROOT/.benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/benchmark_$TIMESTAMP.json"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${CYAN}======================================${NC}"
echo -e "${CYAN}Nanolang Performance Benchmark${NC}"
echo -e "${CYAN}======================================${NC}"
echo ""

mkdir -p "$RESULTS_DIR"

# Initialize results JSON
cat > "$RESULTS_FILE" << EOF
{
  "timestamp": "$TIMESTAMP",
  "date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "benchmarks": {
EOF

# Function to measure time in milliseconds
time_ms() {
    local start=$(date +%s%N)
    "$@" > /dev/null 2>&1
    local end=$(date +%s%N)
    echo $(( (end - start) / 1000000 ))
}

# Function to measure memory usage (peak RSS in KB)
mem_usage() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        /usr/bin/time -l "$@" 2>&1 | grep "maximum resident set size" | awk '{print $1 / 1024}'
    else
        /usr/bin/time -v "$@" 2>&1 | grep "Maximum resident set size" | awk '{print $6}'
    fi
}

cd "$PROJECT_ROOT"

# Clean build to ensure accurate measurements
echo -e "${YELLOW}Cleaning build artifacts...${NC}"
make clean > /dev/null 2>&1

# Benchmark 1: Stage 1 Build (C Compiler)
echo -e "${GREEN}Benchmark 1: Stage 1 Build Time${NC}"
STAGE1_TIME=$(time_ms make stage1)
echo "  Time: ${STAGE1_TIME}ms"

# Benchmark 2: Stage 2 Build (Self-Hosted Components)
echo -e "${GREEN}Benchmark 2: Stage 2 Build Time${NC}"
STAGE2_TIME=$(time_ms make stage2)
echo "  Time: ${STAGE2_TIME}ms"

# Benchmark 3: Stage 3 Build (Bootstrap Validation)
echo -e "${GREEN}Benchmark 3: Stage 3 Build Time${NC}"
STAGE3_TIME=$(time_ms make stage3)
echo "  Time: ${STAGE3_TIME}ms"

# Benchmark 4: Full Test Suite
echo -e "${GREEN}Benchmark 4: Test Suite Execution${NC}"
TEST_TIME=$(time_ms ./tests/run_all_tests.sh)
echo "  Time: ${TEST_TIME}ms"

# Benchmark 5: Self-Hosting Compilation
echo -e "${GREEN}Benchmark 5: Self-Hosting Parser Compilation${NC}"
SELFHOST_TIME=$(time_ms ./bin/nanoc src_nano/parser.nano -o .benchmark_parser_test)
echo "  Time: ${SELFHOST_TIME}ms"
rm -f .benchmark_parser_test .benchmark_parser_test.c

# Benchmark 6: Simple Example Compilation
echo -e "${GREEN}Benchmark 6: Simple Example Compilation${NC}"
EXAMPLE_TIME=$(time_ms ./bin/nanoc examples/nl_factorial.nano -o .benchmark_factorial_test)
echo "  Time: ${EXAMPLE_TIME}ms"
rm -f .benchmark_factorial_test .benchmark_factorial_test.c

# Benchmark 7: Interpreter Execution
echo -e "${GREEN}Benchmark 7: Interpreter Execution (Factorial)${NC}"
INTERP_TIME=$(time_ms ./bin/nano examples/nl_factorial.nano)
echo "  Time: ${INTERP_TIME}ms"

# Benchmark 8: Memory Usage (Compiler)
echo -e "${GREEN}Benchmark 8: Compiler Memory Usage${NC}"
if [[ "$OSTYPE" == "darwin"* ]]; then
    COMP_MEM=$(/usr/bin/time -l ./bin/nanoc examples/nl_factorial.nano -o .benchmark_mem_test 2>&1 | grep "maximum resident set size" | awk '{print int($1 / 1024)}')
else
    COMP_MEM=$(/usr/bin/time -v ./bin/nanoc examples/nl_factorial.nano -o .benchmark_mem_test 2>&1 | grep "Maximum resident set size" | awk '{print int($6)}')
fi
echo "  Memory: ${COMP_MEM} KB"
rm -f .benchmark_mem_test .benchmark_mem_test.c

# Write results to JSON
cat >> "$RESULTS_FILE" << EOF
    "stage1_build_ms": $STAGE1_TIME,
    "stage2_build_ms": $STAGE2_TIME,
    "stage3_build_ms": $STAGE3_TIME,
    "test_suite_ms": $TEST_TIME,
    "self_hosting_compile_ms": $SELFHOST_TIME,
    "example_compile_ms": $EXAMPLE_TIME,
    "interpreter_exec_ms": $INTERP_TIME,
    "compiler_memory_kb": $COMP_MEM,
    "total_build_ms": $((STAGE1_TIME + STAGE2_TIME + STAGE3_TIME))
  }
}
EOF

# Calculate totals
TOTAL_BUILD=$((STAGE1_TIME + STAGE2_TIME + STAGE3_TIME))

echo ""
echo -e "${CYAN}======================================${NC}"
echo -e "${CYAN}Summary${NC}"
echo -e "${CYAN}======================================${NC}"
echo -e "${GREEN}Total Build Time:${NC} ${TOTAL_BUILD}ms"
echo -e "${GREEN}Test Suite Time:${NC} ${TEST_TIME}ms"
echo -e "${GREEN}Self-Hosting Compile:${NC} ${SELFHOST_TIME}ms"
echo -e "${GREEN}Compiler Memory:${NC} ${COMP_MEM} KB"
echo ""
echo -e "${YELLOW}Results saved to:${NC} $RESULTS_FILE"

# Generate comparison with previous run if available
PREV_RESULT=$(ls -t "$RESULTS_DIR"/benchmark_*.json 2>/dev/null | head -2 | tail -1)
if [ -f "$PREV_RESULT" ] && [ "$PREV_RESULT" != "$RESULTS_FILE" ]; then
    echo ""
    echo -e "${CYAN}======================================${NC}"
    echo -e "${CYAN}Comparison with Previous Run${NC}"
    echo -e "${CYAN}======================================${NC}"
    
    # Extract previous values
    PREV_TOTAL=$(grep "total_build_ms" "$PREV_RESULT" | grep -oE '[0-9]+')
    PREV_TEST=$(grep "test_suite_ms" "$PREV_RESULT" | grep -oE '[0-9]+')
    
    # Calculate differences
    DIFF_TOTAL=$((TOTAL_BUILD - PREV_TOTAL))
    DIFF_TEST=$((TEST_TIME - PREV_TEST))
    
    # Show comparison
    if [ $DIFF_TOTAL -lt 0 ]; then
        echo -e "${GREEN}Build Time:${NC} ${DIFF_TOTAL}ms (faster)"
    elif [ $DIFF_TOTAL -gt 0 ]; then
        echo -e "${YELLOW}Build Time:${NC} +${DIFF_TOTAL}ms (slower)"
    else
        echo -e "${GREEN}Build Time:${NC} No change"
    fi
    
    if [ $DIFF_TEST -lt 0 ]; then
        echo -e "${GREEN}Test Time:${NC} ${DIFF_TEST}ms (faster)"
    elif [ $DIFF_TEST -gt 0 ]; then
        echo -e "${YELLOW}Test Time:${NC} +${DIFF_TEST}ms (slower)"
    else
        echo -e "${GREEN}Test Time:${NC} No change"
    fi
fi

echo ""
echo -e "${GREEN}✅ Benchmarking complete!${NC}"

