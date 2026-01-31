#!/bin/bash
# Differential testing: Compare C vs NanoLang compiler outputs
#
# Usage: ./tools/differential_test.sh [test_file.nano]
#        ./tools/differential_test.sh --all      # Run all tests
#        ./tools/differential_test.sh --random N # Generate N random tests

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TEMP_DIR="/tmp/nanolang_diff_test_$$"
C_COMPILER="$ROOT_DIR/bin/nanoc_c"
NANO_COMPILER="$ROOT_DIR/bin/nanoc"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

mkdir -p "$TEMP_DIR"
trap "rm -rf $TEMP_DIR" EXIT

# Statistics
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

log_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((PASSED_TESTS++))
}

log_fail() {
    echo -e "${RED}✗${NC} $1"
    ((FAILED_TESTS++))
}

log_info() {
    echo -e "${YELLOW}ℹ${NC} $1"
}

# Test a single file with both compilers
test_file() {
    local test_file="$1"
    local test_name=$(basename "$test_file" .nano)
    
    ((TOTAL_TESTS++))
    
    log_info "Testing: $test_name"
    
    # Compile with C compiler
    local c_output="$TEMP_DIR/${test_name}_c.c"
    local c_bin="$TEMP_DIR/${test_name}_c"
    if ! timeout 30 "$C_COMPILER" "$test_file" -o "$c_bin" > "$TEMP_DIR/c_compile.log" 2>&1; then
        log_fail "$test_name: C compiler failed"
        cat "$TEMP_DIR/c_compile.log"
        return 1
    fi
    
    # Compile with NanoLang compiler (if available)
    if [ ! -f "$NANO_COMPILER" ]; then
        log_info "$test_name: NanoLang compiler not built, skipping comparison"
        log_pass "$test_name: C compiler succeeded"
        return 0
    fi
    
    local nano_output="$TEMP_DIR/${test_name}_nano.c"
    local nano_bin="$TEMP_DIR/${test_name}_nano"
    if ! timeout 30 "$NANO_COMPILER" "$test_file" -o "$nano_bin" > "$TEMP_DIR/nano_compile.log" 2>&1; then
        log_fail "$test_name: NanoLang compiler failed"
        cat "$TEMP_DIR/nano_compile.log"
        return 1
    fi
    
    # Run both binaries and compare outputs
    local c_run_output="$TEMP_DIR/c_run.txt"
    local nano_run_output="$TEMP_DIR/nano_run.txt"
    
    if ! timeout 5 "$c_bin" > "$c_run_output" 2>&1; then
        local c_exit=$?
        log_fail "$test_name: C binary failed (exit $c_exit)"
        return 1
    fi
    local c_exit=$?
    
    if ! timeout 5 "$nano_bin" > "$nano_run_output" 2>&1; then
        local nano_exit=$?
        log_fail "$test_name: NanoLang binary failed (exit $nano_exit)"
        return 1
    fi
    local nano_exit=$?
    
    # Compare exit codes
    if [ "$c_exit" != "$nano_exit" ]; then
        log_fail "$test_name: Exit codes differ (C: $c_exit, Nano: $nano_exit)"
        return 1
    fi
    
    # Compare outputs
    if ! diff -q "$c_run_output" "$nano_run_output" > /dev/null 2>&1; then
        log_fail "$test_name: Outputs differ"
        echo "=== C output ==="
        cat "$c_run_output"
        echo "=== NanoLang output ==="
        cat "$nano_run_output"
        echo "=== Diff ==="
        diff "$c_run_output" "$nano_run_output" || true
        return 1
    fi
    
    log_pass "$test_name: Both compilers produced identical results"
    return 0
}

# Run all tests in tests/ directory
run_all_tests() {
    log_info "Running differential tests on all test files..."
    
    for test_file in "$ROOT_DIR"/tests/test_*.nano; do
        if [ -f "$test_file" ]; then
            test_file "$test_file" || true
        fi
    done
    
    # Also test examples
    for test_file in "$ROOT_DIR"/examples/*.nano; do
        if [ -f "$test_file" ]; then
            # Skip examples that require user interaction or external dependencies
            local basename=$(basename "$test_file")
            case "$basename" in
                *sdl*|*ncurses*|*glfw*|*opengl*|*audio*|*mixer*)
                    log_info "Skipping interactive/dependency example: $basename"
                    continue
                    ;;
            esac
            test_file "$test_file" || true
        fi
    done
}

# Print summary
print_summary() {
    echo ""
    echo "========================================"
    echo "DIFFERENTIAL TESTING SUMMARY"
    echo "========================================"
    echo "Total tests:  $TOTAL_TESTS"
    echo -e "Passed:       ${GREEN}$PASSED_TESTS${NC}"
    echo -e "Failed:       ${RED}$FAILED_TESTS${NC}"
    
    if [ $FAILED_TESTS -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        echo "========================================"
        return 0
    else
        local pass_rate=$((PASSED_TESTS * 100 / TOTAL_TESTS))
        echo "Pass rate:    $pass_rate%"
        echo "========================================"
        return 1
    fi
}

# Main
main() {
    if [ $# -eq 0 ]; then
        echo "Usage: $0 [test_file.nano]"
        echo "       $0 --all"
        echo "       $0 --random N"
        exit 1
    fi
    
    case "$1" in
        --all)
            run_all_tests
            print_summary
            ;;
        --random)
            echo "Random test generation not yet implemented"
            echo "TODO: Implement fuzzer to generate random valid NanoLang programs"
            exit 1
            ;;
        *)
            if [ -f "$1" ]; then
                test_file "$1"
                print_summary
            else
                echo "Error: File not found: $1"
                exit 1
            fi
            ;;
    esac
}

main "$@"

