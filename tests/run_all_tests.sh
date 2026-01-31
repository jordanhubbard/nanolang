#!/usr/bin/env bash
# ============================================================================
# Nanolang Comprehensive Test Suite
# ============================================================================
# 
# Test Categories:
#   nl_*     - Core Language Tests (syntax, types, control flow, functions)
#   app_*    - Application/Integration Tests (higher-level compositions)
#   neg_*    - Negative Tests (expected failures)
#   unit/*   - Unit Tests (comprehensive feature tests)
#
# Usage:
#   ./tests/run_all_tests.sh           # Run all tests
#   ./tests/run_all_tests.sh --lang    # Run only language tests (nl_*)
#   ./tests/run_all_tests.sh --app     # Run only application tests
#   ./tests/run_all_tests.sh --unit    # Run only unit tests
#
# ============================================================================

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Counters
TOTAL_PASS=0
TOTAL_FAIL=0
TOTAL_SKIP=0

# Category counters
NL_PASS=0
NL_FAIL=0
APP_PASS=0
APP_FAIL=0
UNIT_PASS=0
UNIT_FAIL=0

# Parse arguments
RUN_LANG=true
RUN_APP=true
RUN_UNIT=true

if [ "$1" = "--lang" ]; then
    RUN_APP=false
    RUN_UNIT=false
elif [ "$1" = "--app" ]; then
    RUN_LANG=false
    RUN_UNIT=false
elif [ "$1" = "--unit" ]; then
    RUN_LANG=false
    RUN_APP=false
fi

# Expected failures (features not fully implemented)
# None currently - function variables are now fully supported!
EXPECTED_FAILURES=(
)

is_expected_failure() {
    local test_name="$1"
    for exp in "${EXPECTED_FAILURES[@]}"; do
        if [ "$test_name" = "$exp" ]; then
            return 0
        fi
    done
    return 1
}

# Run a single test
run_test() {
    local test_file="$1"
    local test_name=$(basename "$test_file")
    local category="$2"
    
    # Check for expected failures
    if is_expected_failure "$test_name"; then
        echo -e "${BLUE}⊘${NC} $test_name ${YELLOW}(expected failure)${NC}"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return 0
    fi
    
    # Run the test
    local result=$(./bin/nanoc "$test_file" 2>&1)
    
    if echo "$result" | grep -q "All shadow tests passed"; then
        echo -e "${GREEN}✅${NC} $test_name"
        TOTAL_PASS=$((TOTAL_PASS + 1))
        case "$category" in
            "nl") NL_PASS=$((NL_PASS + 1)) ;;
            "app") APP_PASS=$((APP_PASS + 1)) ;;
            "unit") UNIT_PASS=$((UNIT_PASS + 1)) ;;
        esac
        return 0
    else
        echo -e "${RED}❌${NC} $test_name"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        case "$category" in
            "nl") NL_FAIL=$((NL_FAIL + 1)) ;;
            "app") APP_FAIL=$((APP_FAIL + 1)) ;;
            "unit") UNIT_FAIL=$((UNIT_FAIL + 1)) ;;
        esac
        return 1
    fi
}

echo ""
echo -e "${BOLD}========================================"
echo "NANOLANG COMPREHENSIVE TEST SUITE"
echo -e "========================================${NC}"
echo ""

# ============================================================================
# CATEGORY 1: Core Language Tests (nl_*)
# ============================================================================
if [ "$RUN_LANG" = true ]; then
    echo -e "${CYAN}=== CORE LANGUAGE TESTS (nl_*) ===${NC}"
    echo ""
    
    # Syntax tests
    echo -e "${BOLD}-- Syntax --${NC}"
    for f in tests/nl_syntax_*.nano; do
        [ -f "$f" ] && run_test "$f" "nl"
    done
    echo ""
    
    # Type tests
    echo -e "${BOLD}-- Types --${NC}"
    for f in tests/nl_types_*.nano; do
        [ -f "$f" ] && run_test "$f" "nl"
    done
    echo ""
    
    # Control flow tests
    echo -e "${BOLD}-- Control Flow --${NC}"
    for f in tests/nl_control_*.nano; do
        [ -f "$f" ] && run_test "$f" "nl"
    done
    echo ""
    
    # Function tests
    echo -e "${BOLD}-- Functions --${NC}"
    for f in tests/nl_functions_*.nano; do
        [ -f "$f" ] && run_test "$f" "nl"
    done
    echo ""
    
    echo -e "${CYAN}Language Tests: ${GREEN}$NL_PASS passed${NC}, ${RED}$NL_FAIL failed${NC}"
    echo ""
fi

# ============================================================================
# CATEGORY 2: Application/Integration Tests
# ============================================================================
if [ "$RUN_APP" = true ]; then
    echo -e "${CYAN}=== APPLICATION TESTS ===${NC}"
    echo ""
    
    # Tuple tests (comprehensive application tests)
    echo -e "${BOLD}-- Tuple Applications --${NC}"
    for f in tests/tuple_*.nano; do
        [ -f "$f" ] && run_test "$f" "app"
    done
    echo ""
    
    # Nested struct tests
    echo -e "${BOLD}-- Nested Structures --${NC}"
    for f in tests/nested_*.nano tests/test_nested_*.nano; do
        [ -f "$f" ] && run_test "$f" "app"
    done
    echo ""
    
    # Other application tests
    echo -e "${BOLD}-- Other Applications --${NC}"
    for f in tests/test_*.nano; do
        name=$(basename "$f")
        # Skip tests already covered or that are unit tests
        case "$name" in
            test_nested_*|nl_*|test_firstclass_*|test_unions_match_*) continue ;;
        esac
        [ -f "$f" ] && run_test "$f" "app"
    done
    echo ""
    
    echo -e "${CYAN}Application Tests: ${GREEN}$APP_PASS passed${NC}, ${RED}$APP_FAIL failed${NC}"
    echo ""
fi

# ============================================================================
# CATEGORY 3: Unit Tests (comprehensive feature tests)
# ============================================================================
if [ "$RUN_UNIT" = true ]; then
    echo -e "${CYAN}=== UNIT TESTS ===${NC}"
    echo ""
    
    for f in tests/unit/*.nano; do
        [ -f "$f" ] && run_test "$f" "unit"
    done
    echo ""
    
    echo -e "${CYAN}Unit Tests: ${GREEN}$UNIT_PASS passed${NC}, ${RED}$UNIT_FAIL failed${NC}"
    echo ""
fi

# ============================================================================
# Summary
# ============================================================================
echo -e "${BOLD}========================================"
echo "TEST RESULTS SUMMARY"
echo -e "========================================${NC}"
echo ""

if [ "$RUN_LANG" = true ]; then
    echo -e "Core Language (nl_*):  ${GREEN}$NL_PASS passed${NC}, ${RED}$NL_FAIL failed${NC}"
fi
if [ "$RUN_APP" = true ]; then
    echo -e "Application Tests:     ${GREEN}$APP_PASS passed${NC}, ${RED}$APP_FAIL failed${NC}"
fi
if [ "$RUN_UNIT" = true ]; then
    echo -e "Unit Tests:            ${GREEN}$UNIT_PASS passed${NC}, ${RED}$UNIT_FAIL failed${NC}"
fi

echo ""
echo -e "TOTAL: ${GREEN}$TOTAL_PASS passed${NC}, ${RED}$TOTAL_FAIL failed${NC}, ${BLUE}$TOTAL_SKIP skipped${NC}"
echo ""

if [ $TOTAL_FAIL -eq 0 ]; then
    echo -e "${GREEN}✅ All runnable tests passed!${NC}"
    if [ $TOTAL_SKIP -gt 0 ]; then
        echo -e "${YELLOW}   ($TOTAL_SKIP skipped due to unimplemented features)${NC}"
    fi
    exit 0
else
    echo -e "${RED}❌ Some tests failed${NC}"
    exit 1
fi
