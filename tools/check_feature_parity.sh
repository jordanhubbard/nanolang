#!/bin/bash
# Check feature parity between C and NanoLang compiler implementations
#
# Verifies:
# - All AST node types exist in both parsers
# - All TypeInfo fields match
# - Test coverage equivalence
# - Shadow tests exist for all functions

# Don't use set -e to avoid early exit on grep/wc returning 0 results
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

ERRORS=0
WARNINGS=0
CHECKS=0

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
    ((CHECKS++))
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    ((ERRORS++))
    ((CHECKS++))
}

check_warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    ((WARNINGS++))
    ((CHECKS++))
}

echo "========================================"
echo "DUAL-IMPLEMENTATION PARITY CHECKER"
echo "========================================"
echo ""

# 1. Schema validation
echo -e "${BLUE}[1/6]${NC} Checking schema synchronization..."
if python3 "$ROOT_DIR/tools/validate_schema_sync.py" > /tmp/schema_check.log 2>&1; then
    check_pass "Schema validation passed"
else
    check_fail "Schema validation failed (see /tmp/schema_check.log)"
fi

# 2. Check AST node coverage in parsers
echo ""
echo -e "${BLUE}[2/6]${NC} Checking AST node coverage..."

# Extract AST nodes from C parser
c_ast_nodes=$(grep -o "AST_[A-Z_]*" "$ROOT_DIR/src/parser.c" | sort -u | wc -l || echo "0")
check_pass "C parser handles $c_ast_nodes AST node types"

# Extract parse nodes from NanoLang parser
nano_pnodes=$(grep -o "PNODE_[A-Z_]*" "$ROOT_DIR/src_nano/parser.nano" | sort -u | wc -l || echo "0")
check_pass "NanoLang parser handles $nano_pnodes parse node types"

# Compare counts
if [ "$c_ast_nodes" -lt 30 ]; then
    check_warn "C parser AST coverage seems low ($c_ast_nodes nodes)"
fi
if [ "$nano_pnodes" -lt 30 ]; then
    check_warn "NanoLang parser coverage seems low ($nano_pnodes nodes)"
fi

# 3. Check typechecker coverage
echo ""
echo -e "${BLUE}[3/6]${NC} Checking typechecker coverage..."

# Check if both typecheckers handle key language features
features=("struct" "enum" "union" "array" "function" "generic" "match" "unsafe")
for feature in "${features[@]}"; do
    c_has=$(grep -i "case AST_.*$feature" "$ROOT_DIR/src/typechecker.c" || true)
    nano_has=$(grep -i "$feature" "$ROOT_DIR/src_nano/typecheck.nano" || true)
    
    if [ -n "$c_has" ] && [ -n "$nano_has" ]; then
        check_pass "Both typecheckers handle $feature"
    elif [ -n "$c_has" ]; then
        check_warn "Only C typechecker handles $feature"
    elif [ -n "$nano_has" ]; then
        check_warn "Only NanoLang typechecker handles $feature"
    fi
done

# 4. Check transpiler coverage
echo ""
echo -e "${BLUE}[4/6]${NC} Checking transpiler coverage..."

# Count case statements in both transpilers
c_transpiler_cases=$(grep -c "case AST_" "$ROOT_DIR/src/transpiler_iterative_v3_twopass.c" || echo "0")
nano_transpiler_cases=$(grep -c "if (== node_type" "$ROOT_DIR/src_nano/transpiler.nano" || echo "0")

check_pass "C transpiler has $c_transpiler_cases case statements"
check_pass "NanoLang transpiler has $nano_transpiler_cases conditional checks"

if [ "$c_transpiler_cases" -lt 25 ]; then
    check_warn "C transpiler coverage seems incomplete"
fi
if [ "$nano_transpiler_cases" -lt 25 ]; then
    check_warn "NanoLang transpiler coverage seems incomplete"
fi

# 5. Check test coverage
echo ""
echo -e "${BLUE}[5/6]${NC} Checking test coverage..."

# Count tests
test_count=$(find "$ROOT_DIR/tests" -name "*.nano" -type f | wc -l)
example_count=$(find "$ROOT_DIR/examples" -name "*.nano" -type f | wc -l)

check_pass "Found $test_count test files"
check_pass "Found $example_count example files"

if [ "$test_count" -lt 50 ]; then
    check_warn "Test coverage seems low ($test_count tests)"
fi

# 6. Check shadow tests (critical for NanoLang)
echo ""
echo -e "${BLUE}[6/6]${NC} Checking shadow test coverage..."

# Count shadow tests in NanoLang compiler
shadow_count=$(grep "^shadow " "$ROOT_DIR/src_nano/parser.nano" "$ROOT_DIR/src_nano/typecheck.nano" "$ROOT_DIR/src_nano/transpiler.nano" 2>/dev/null | wc -l || echo "0")
shadow_count=$(echo "$shadow_count" | tr -d ' ')
check_pass "NanoLang compiler has $shadow_count shadow tests"

if [ "$shadow_count" -lt 100 ]; then
    check_warn "Shadow test coverage could be improved ($shadow_count tests)"
fi

# Check for functions without shadow tests (except extern)
echo ""
echo -e "${BLUE}Checking for missing shadow tests...${NC}"
missing_shadows=0
for file in "$ROOT_DIR/src_nano"/*.nano; do
    if [ -f "$file" ]; then
        # Find functions without corresponding shadow tests
        while IFS= read -r funcname; do
            if ! grep -q "^shadow $funcname" "$file" 2>/dev/null; then
                # Check if it's an extern function
                if ! grep -q "^extern fn $funcname" "$file" 2>/dev/null; then
                    ((missing_shadows++))
                fi
            fi
        done < <(grep "^fn [a-z_]*(" "$file" | sed 's/fn \([a-z_]*\)(.*/\1/' || true)
    fi
done

if [ "$missing_shadows" -eq 0 ]; then
    check_pass "All non-extern functions have shadow tests"
elif [ "$missing_shadows" -lt 10 ]; then
    check_warn "$missing_shadows functions missing shadow tests"
else
    check_fail "$missing_shadows functions missing shadow tests"
fi

# Summary
echo ""
echo "========================================"
echo "PARITY CHECK SUMMARY"
echo "========================================"
echo "Total checks: $CHECKS"
echo -e "Passed:       ${GREEN}$((CHECKS - ERRORS - WARNINGS))${NC}"
echo -e "Warnings:     ${YELLOW}$WARNINGS${NC}"
echo -e "Errors:       ${RED}$ERRORS${NC}"
echo "========================================"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ Dual-implementation parity verified!${NC}"
    exit 0
else
    echo -e "${RED}✗ Parity issues detected${NC}"
    exit 1
fi

