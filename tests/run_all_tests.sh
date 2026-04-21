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
# Backend selection (env var):
#   NANOLANG_BACKEND=c       ./tests/run_all_tests.sh   # C transpiler (default)
#   NANOLANG_BACKEND=vm      ./tests/run_all_tests.sh   # NanoVM bytecode
#   NANOLANG_BACKEND=daemon  ./tests/run_all_tests.sh   # NanoVM daemon mode
#
# ============================================================================

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Backend selection: c (default), vm, daemon
BACKEND="${NANOLANG_BACKEND:-c}"

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

# Structured output format: "" (default human), "junit", "tap"
OUTPUT_FORMAT=""
OUTPUT_FILE=""

# Parse all arguments (order-independent)
for arg in "$@"; do
    case "$arg" in
        --lang)    RUN_APP=false; RUN_UNIT=false ;;
        --app)     RUN_LANG=false; RUN_UNIT=false ;;
        --unit)    RUN_LANG=false; RUN_APP=false ;;
        --format=junit) OUTPUT_FORMAT="junit" ;;
        --format=tap)   OUTPUT_FORMAT="tap" ;;
        --format=*) echo "Unknown format: ${arg#--format=} (supported: junit, tap)" >&2; exit 1 ;;
        --output=*) OUTPUT_FILE="${arg#--output=}" ;;
        --help|-h)
            echo "Usage: $0 [--lang|--app|--unit] [--format=junit|tap] [--output=FILE]"
            exit 0 ;;
    esac
done

# Default output file by format
if [ -n "$OUTPUT_FORMAT" ] && [ -z "$OUTPUT_FILE" ]; then
    case "$OUTPUT_FORMAT" in
        junit) OUTPUT_FILE="test-results.xml" ;;
        tap)   OUTPUT_FILE="test-results.tap" ;;
    esac
fi

# Arrays to accumulate structured test results (parallel arrays)
RESULT_NAMES=()
RESULT_STATUS=()   # "pass" | "fail" | "skip"
RESULT_TIMES=()    # elapsed seconds (float)
RESULT_MESSAGES=() # failure message if any
RESULT_CLASSES=()  # category: nl / app / unit

# Wall-clock start for timing
SUITE_START=$(date +%s%N 2>/dev/null || date +%s)

# Helper: record a test result
record_result() {
    local name="$1" status="$2" elapsed="$3" message="$4" class="$5"
    RESULT_NAMES+=("$name")
    RESULT_STATUS+=("$status")
    RESULT_TIMES+=("$elapsed")
    RESULT_MESSAGES+=("$message")
    RESULT_CLASSES+=("$class")
}

# Expected failures (features not fully implemented)
# None currently - function variables are now fully supported!
EXPECTED_FAILURES=(
    # Negative tests: expected to fail compilation (semantic errors in algebraic effects)
    test_effects_negative.nano
    test_effects_neg2.nano
    # These tests require unmerged feature branches and compile against stub implementations.
    # They will be un-skipped as the corresponding branches land in main.
    test_coroutine.nano      # requires feat/coroutine-runtime
    test_pretty_printer.nano # requires feat/nano-fmt or similar formatting branch
)

# Per-backend expected failures
VM_EXPECTED_FAILURES=(
    # Add tests here only for features that are INTENTIONALLY unsupported,
    # not for gaps that should be fixed.
)

DAEMON_EXPECTED_FAILURES=(
    # get_argc() returns 0 in co-process isolation (no command-line context)
    ug_patterns_unsafe_extern.nano
)

is_expected_failure() {
    local test_name="$1"
    for exp in "${EXPECTED_FAILURES[@]}"; do
        if [ "$test_name" = "$exp" ]; then
            return 0
        fi
    done
    # Check backend-specific expected failures
    case "$BACKEND" in
        vm|nanovm)
            for exp in "${VM_EXPECTED_FAILURES[@]}"; do
                if [ "$test_name" = "$exp" ]; then
                    return 0
                fi
            done
            ;;
        daemon)
            for exp in "${DAEMON_EXPECTED_FAILURES[@]}"; do
                if [ "$test_name" = "$exp" ]; then
                    return 0
                fi
            done
            ;;
    esac
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
        record_result "$test_name" "skip" "0" "expected failure" "$category"
        return 0
    fi

    local _t0
    _t0=$(date +%s%N 2>/dev/null || date +%s)
    
    local log_file=".test_output/${category}_${test_name}.compile.log"
    local out_file=".test_output/${category}_${test_name}.out"
    local run_log=".test_output/${category}_${test_name}.run.log"
    
    # Compile the test (backend-dependent)
    case "$BACKEND" in
        c|native)
            perl -e "alarm $COMPILE_TIMEOUT; exec @ARGV" ./bin/nanoc "$test_file" -o "$out_file" >"$log_file" 2>&1
            ;;
        vm|nanovm)
            perl -e "alarm $COMPILE_TIMEOUT; exec @ARGV" ./bin/nano_virt "$test_file" --emit-nvm -o "${out_file}.nvm" >"$log_file" 2>&1
            ;;
        daemon)
            perl -e "alarm $COMPILE_TIMEOUT; exec @ARGV" ./bin/nano_virt "$test_file" --emit-nvm -o "${out_file}.nvm" >"$log_file" 2>&1
            ;;
        *)
            echo -e "${RED}❌${NC} $test_name ${RED}(unknown backend: $BACKEND)${NC}"
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
            return 1
            ;;
    esac

    local _compile_exit=$?
    if [ $_compile_exit -ne 0 ]; then
        local _elapsed_val
        _elapsed_val=$(_elapsed)
        local _compile_err
        _compile_err=$(cat "$log_file" 2>/dev/null | head -5 | tr '\n' ' ' | sed 's/[<>&"]/./g')
        echo -e "${RED}❌${NC} $test_name ${RED}(compilation failed)${NC}"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        case "$category" in
            "nl") NL_FAIL=$((NL_FAIL + 1)) ;;
            "app") APP_FAIL=$((APP_FAIL + 1)) ;;
            "unit") UNIT_FAIL=$((UNIT_FAIL + 1)) ;;
        esac
        record_result "$test_name" "fail" "$_elapsed_val" "compilation failed: $_compile_err" "$category"
        return 1
    fi

    # Determine the artifact to check and run command
    local run_artifact run_cmd
    case "$BACKEND" in
        c|native)
            run_artifact="$out_file"
            run_cmd="$out_file"
            ;;
        vm|nanovm)
            run_artifact="${out_file}.nvm"
            run_cmd="./bin/nano_vm ${out_file}.nvm"
            ;;
        daemon)
            run_artifact="${out_file}.nvm"
            run_cmd="./bin/nano_vm --daemon ${out_file}.nvm"
            ;;
    esac

    # Helper: compute elapsed time in seconds (float, 2 decimal places)
    _elapsed() {
        local t1
        t1=$(date +%s%N 2>/dev/null || date +%s)
        # If nanoseconds available, divide; otherwise just integer seconds
        if echo "$t1" | grep -q '[0-9]\{10,\}'; then
            echo "scale=2; ($t1 - $_t0) / 1000000000" | bc 2>/dev/null || echo "0"
        else
            echo "$((t1 - _t0))"
        fi
    }

    # Run the compiled artifact
    if [ -f "$run_artifact" ]; then
        perl -e "alarm $RUN_TIMEOUT; exec @ARGV" $run_cmd >"$run_log" 2>&1
        local _run_exit=$?
        local _elapsed_val
        _elapsed_val=$(_elapsed)
        if [ $_run_exit -eq 0 ]; then
            echo -e "${GREEN}✅${NC} $test_name"
            rm -f "$log_file" "$out_file" "${out_file}.nvm" "$run_log"
            TOTAL_PASS=$((TOTAL_PASS + 1))
            case "$category" in
                "nl") NL_PASS=$((NL_PASS + 1)) ;;
                "app") APP_PASS=$((APP_PASS + 1)) ;;
                "unit") UNIT_PASS=$((UNIT_PASS + 1)) ;;
            esac
            record_result "$test_name" "pass" "$_elapsed_val" "" "$category"
            return 0
        else
            local _fail_msg
            _fail_msg=$(cat "$run_log" 2>/dev/null | head -5 | tr '\n' ' ' | sed 's/[<>&"]/./g')
            echo -e "${RED}❌${NC} $test_name ${RED}(runtime failure)${NC}"
            TOTAL_FAIL=$((TOTAL_FAIL + 1))
            case "$category" in
                "nl") NL_FAIL=$((NL_FAIL + 1)) ;;
                "app") APP_FAIL=$((APP_FAIL + 1)) ;;
                "unit") UNIT_FAIL=$((UNIT_FAIL + 1)) ;;
            esac
            record_result "$test_name" "fail" "$_elapsed_val" "runtime failure: $_fail_msg" "$category"
            return 1
        fi
    else
        local _elapsed_val
        _elapsed_val=$(_elapsed)
        local _compile_msg
        _compile_msg=$(cat "$log_file" 2>/dev/null | head -5 | tr '\n' ' ' | sed 's/[<>&"]/./g')
        echo -e "${RED}❌${NC} $test_name ${RED}(output not created)${NC}"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
        case "$category" in
            "nl") NL_FAIL=$((NL_FAIL + 1)) ;;
            "app") APP_FAIL=$((APP_FAIL + 1)) ;;
            "unit") UNIT_FAIL=$((UNIT_FAIL + 1)) ;;
        esac
        record_result "$test_name" "fail" "$_elapsed_val" "compile failure: $_compile_msg" "$category"
        return 1
    fi
}

echo ""
echo -e "${BOLD}========================================"
echo "NANOLANG COMPREHENSIVE TEST SUITE"
echo -e "========================================${NC}"
echo -e "Backend: ${CYAN}${BACKEND}${NC}"
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
# These tests are C-transpiler-specific (--llm-diags-json, --llm-diags-toon)
# ============================================================================
if [ "$RUN_APP" = true ] && [ "$BACKEND" = "c" -o "$BACKEND" = "native" ]; then
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

# ── Structured output emission ────────────────────────────────────────────────

emit_junit() {
    local outfile="$1"
    local suite_time
    local _t1
    _t1=$(date +%s%N 2>/dev/null || date +%s)
    if echo "$_t1" | grep -q '[0-9]\{10,\}'; then
        suite_time=$(echo "scale=2; ($_t1 - $SUITE_START) / 1000000000" | bc 2>/dev/null || echo "0")
    else
        suite_time=$(($(date +%s) - SUITE_START))
    fi

    local total=${#RESULT_NAMES[@]}
    local failures=$TOTAL_FAIL
    local skipped=$TOTAL_SKIP
    local timestamp
    timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    {
        echo '<?xml version="1.0" encoding="UTF-8"?>'
        echo "<testsuites name=\"nanolang\" tests=\"$total\" failures=\"$failures\" skipped=\"$skipped\" time=\"$suite_time\" timestamp=\"$timestamp\">"
        echo "  <testsuite name=\"nanolang\" tests=\"$total\" failures=\"$failures\" skipped=\"$skipped\" time=\"$suite_time\" hostname=\"$(hostname)\">"
        for i in "${!RESULT_NAMES[@]}"; do
            local name="${RESULT_NAMES[$i]}"
            local status="${RESULT_STATUS[$i]}"
            local elapsed="${RESULT_TIMES[$i]:-0}"
            local message="${RESULT_MESSAGES[$i]}"
            local class="${RESULT_CLASSES[$i]:-nanolang}"
            # Sanitise name: strip .nano suffix, replace / with .
            local classname="nanolang.${class}.$(echo "$name" | sed 's/\.nano$//' | tr '/' '.')"
            local testname
            testname=$(echo "$name" | sed 's/\.nano$//')
            echo "    <testcase classname=\"$classname\" name=\"$testname\" time=\"$elapsed\">"
            case "$status" in
                fail)
                    local msg_escaped
                    msg_escaped=$(echo "$message" | sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g; s/"/\&quot;/g')
                    echo "      <failure message=\"$msg_escaped\" type=\"TestFailure\">$msg_escaped</failure>"
                    ;;
                skip)
                    echo "      <skipped/>"
                    ;;
            esac
            echo "    </testcase>"
        done
        echo "  </testsuite>"
        echo "</testsuites>"
    } > "$outfile"
    echo -e "${CYAN}📋 JUnit XML written to: $outfile${NC}"
}

emit_tap() {
    local outfile="$1"
    local total=${#RESULT_NAMES[@]}
    {
        echo "TAP version 13"
        echo "1..$total"
        for i in "${!RESULT_NAMES[@]}"; do
            local name="${RESULT_NAMES[$i]}"
            local status="${RESULT_STATUS[$i]}"
            local message="${RESULT_MESSAGES[$i]}"
            local testname
            testname=$(echo "$name" | sed 's/\.nano$//')
            local n=$((i + 1))
            case "$status" in
                pass) echo "ok $n - $testname" ;;
                fail) echo "not ok $n - $testname"
                      if [ -n "$message" ]; then
                          echo "  ---"
                          echo "  message: '$message'"
                          echo "  ..."
                      fi ;;
                skip) echo "ok $n - $testname # SKIP expected failure" ;;
            esac
        done
    } > "$outfile"
    echo -e "${CYAN}📋 TAP output written to: $outfile${NC}"
}

# Emit structured output if requested
if [ -n "$OUTPUT_FORMAT" ]; then
    case "$OUTPUT_FORMAT" in
        junit) emit_junit "$OUTPUT_FILE" ;;
        tap)   emit_tap   "$OUTPUT_FILE" ;;
    esac
fi

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
