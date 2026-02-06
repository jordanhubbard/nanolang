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

mkdir -p .test_output
rm -f .test_output/*.compile.log

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

    # Per-test timeouts (seconds). These prevent a single compiler hang from stalling the suite.
    # Override via env if needed (e.g. CI).
    local COMPILE_TIMEOUT="${NANOLANG_TEST_COMPILE_TIMEOUT:-60}"
    local RUN_TIMEOUT="${NANOLANG_TEST_RUN_TIMEOUT:-60}"
    
    # Check for expected failures
    if is_expected_failure "$test_name"; then
        echo -e "${BLUE}⊘${NC} $test_name ${YELLOW}(expected failure)${NC}"
        TOTAL_SKIP=$((TOTAL_SKIP + 1))
        return 0
    fi
    
    local log_file=".test_output/${category}_${test_name}.compile.log"
    local out_file=".test_output/${category}_${test_name}.out"
    local run_log=".test_output/${category}_${test_name}.run.log"
    
    # Compile the test
    perl -e "alarm $COMPILE_TIMEOUT; exec @ARGV" ./bin/nanoc "$test_file" -o "$out_file" >"$log_file" 2>&1
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌${NC} $test_name ${RED}(compilation failed)${NC}"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        case "$category" in
            "nl") NL_FAIL=$((NL_FAIL + 1)) ;;
            "app") APP_FAIL=$((APP_FAIL + 1)) ;;
            "unit") UNIT_FAIL=$((UNIT_FAIL + 1)) ;;
        esac
        return 1
    fi
    
    # Run the compiled binary (shadow tests execute at runtime now)
    if [ -f "$out_file" ]; then
        perl -e "alarm $RUN_TIMEOUT; exec @ARGV" "$out_file" >"$run_log" 2>&1
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅${NC} $test_name"
            rm -f "$log_file" "$out_file" "$run_log"
            TOTAL_PASS=$((TOTAL_PASS + 1))
            case "$category" in
                "nl") NL_PASS=$((NL_PASS + 1)) ;;
                "app") APP_PASS=$((APP_PASS + 1)) ;;
                "unit") UNIT_PASS=$((UNIT_PASS + 1)) ;;
            esac
            return 0
        else
            echo -e "${RED}❌${NC} $test_name ${RED}(runtime failure)${NC}"
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
            case "$category" in
                "nl") NL_FAIL=$((NL_FAIL + 1)) ;;
                "app") APP_FAIL=$((APP_FAIL + 1)) ;;
                "unit") UNIT_FAIL=$((UNIT_FAIL + 1)) ;;
            esac
            return 1
        fi
    else
        echo -e "${RED}❌${NC} $test_name ${RED}(binary not created)${NC}"
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
# CATEGORY 3: User Guide Snippets
# ============================================================================
if [ "$RUN_APP" = true ] && [ -d tests/user_guide ]; then
    echo -e "${CYAN}=== USER GUIDE SNIPPETS ===${NC}"
    echo ""
    for f in tests/user_guide/*.nano; do
        [ -f "$f" ] && run_test "$f" "app"
    done
    echo ""
    echo -e "${CYAN}User Guide Snippets: ${GREEN}$APP_PASS passed${NC}, ${RED}$APP_FAIL failed${NC}"
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
# CATEGORY 4: Integration Tests (compiler flags, tooling)
# ============================================================================
if [ "$RUN_APP" = true ]; then
    echo -e "${CYAN}=== INTEGRATION TESTS ===${NC}"
    echo ""
    
    INTEG_PASS=0
    INTEG_FAIL=0
    
    # LLM Diagnostic Flags Tests
    INTEG_TEMP=$(mktemp -d)
    trap "rm -rf $INTEG_TEMP" EXIT
    
    # Create test fixtures
    cat > "$INTEG_TEMP/error.nano" << 'FIXTURE'
fn main() -> int {
    return "not an int"
}
FIXTURE
    
    cat > "$INTEG_TEMP/success.nano" << 'FIXTURE'
fn main() -> int {
    return 0
}

shadow main {
    assert (== (main) 0)
}
FIXTURE
    
    # Test: --llm-diags-json (error case)
    ./bin/nanoc "$INTEG_TEMP/error.nano" --llm-diags-json "$INTEG_TEMP/d.json" >/dev/null 2>&1 || true
    if [ -f "$INTEG_TEMP/d.json" ] && grep -q '"success":false' "$INTEG_TEMP/d.json"; then
        echo -e "${GREEN}✅${NC} --llm-diags-json (error)"
        INTEG_PASS=$((INTEG_PASS + 1))
        TOTAL_PASS=$((TOTAL_PASS + 1))
    else
        echo -e "${RED}❌${NC} --llm-diags-json (error)"
        INTEG_FAIL=$((INTEG_FAIL + 1))
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
    
    # Test: --llm-diags-json (success case)
    ./bin/nanoc "$INTEG_TEMP/success.nano" -o "$INTEG_TEMP/out" --llm-diags-json "$INTEG_TEMP/ok.json" >/dev/null 2>&1
    if [ -f "$INTEG_TEMP/ok.json" ] && grep -q '"success":true' "$INTEG_TEMP/ok.json"; then
        echo -e "${GREEN}✅${NC} --llm-diags-json (success)"
        INTEG_PASS=$((INTEG_PASS + 1))
        TOTAL_PASS=$((TOTAL_PASS + 1))
    else
        echo -e "${RED}❌${NC} --llm-diags-json (success)"
        INTEG_FAIL=$((INTEG_FAIL + 1))
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
    
    # Test: --llm-diags-toon (error case with content verification)
    ./bin/nanoc "$INTEG_TEMP/error.nano" --llm-diags-toon "$INTEG_TEMP/d.toon" >/dev/null 2>&1 || true
    # Verify: file exists, has header with count, has column headers, has data row with error code
    if [ -f "$INTEG_TEMP/d.toon" ] && \
       grep -q 'diagnostics\[1\]:' "$INTEG_TEMP/d.toon" && \
       grep -q 'severity.*code.*message.*file.*line.*column' "$INTEG_TEMP/d.toon" && \
       grep -q 'error.*CTYPE01' "$INTEG_TEMP/d.toon" && \
       grep -q 'diagnostic_count: 1' "$INTEG_TEMP/d.toon"; then
        echo -e "${GREEN}✅${NC} --llm-diags-toon (error)"
        INTEG_PASS=$((INTEG_PASS + 1))
        TOTAL_PASS=$((TOTAL_PASS + 1))
    else
        echo -e "${RED}❌${NC} --llm-diags-toon (error)"
        INTEG_FAIL=$((INTEG_FAIL + 1))
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
    
    # Test: --llm-diags-toon (success case)
    ./bin/nanoc "$INTEG_TEMP/success.nano" -o "$INTEG_TEMP/out2" --llm-diags-toon "$INTEG_TEMP/ok.toon" >/dev/null 2>&1
    if [ -f "$INTEG_TEMP/ok.toon" ] && grep -q 'diagnostic_count: 0' "$INTEG_TEMP/ok.toon"; then
        echo -e "${GREEN}✅${NC} --llm-diags-toon (success)"
        INTEG_PASS=$((INTEG_PASS + 1))
        TOTAL_PASS=$((TOTAL_PASS + 1))
    else
        echo -e "${RED}❌${NC} --llm-diags-toon (success)"
        INTEG_FAIL=$((INTEG_FAIL + 1))
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
    
    rm -rf "$INTEG_TEMP"
    echo ""
    echo -e "${CYAN}Integration Tests: ${GREEN}$INTEG_PASS passed${NC}, ${RED}$INTEG_FAIL failed${NC}"
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
