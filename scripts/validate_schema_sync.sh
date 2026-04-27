#!/usr/bin/env bash
# Validate that C and NanoLang AST/IR definitions match the schema
# Replaces tools/validate_schema_sync.py (Python → Shell for bootstrap)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

# Paths
SCHEMA_PATH="schema/compiler_schema.json"
C_GENERATED="src/generated/compiler_schema.h"
NANO_SCHEMA="src_nano/generated/compiler_schema.nano"
NANO_AST="src_nano/generated/compiler_ast.nano"

# Counters
ERRORS=0
WARNINGS=0
PASSED=0

# Helper: Print error
error() {
    echo -e "${RED}❌ ERROR: $1${NC}" >&2
    ERRORS=$((ERRORS + 1))
}

# Helper: Print warning
warning() {
    echo -e "${YELLOW}⚠️  WARNING: $1${NC}" >&2
    WARNINGS=$((WARNINGS + 1))
}

# Helper: Print success
passed() {
    echo -e "${GREEN}✓ $1${NC}"
    PASSED=$((PASSED + 1))
}

# Extract tokens from schema JSON
extract_schema_tokens() {
    jq -r '.tokens[]' "$SCHEMA_PATH" | sort
}

# Extract parse nodes from schema JSON
extract_schema_parse_nodes() {
    jq -r '.parse_nodes[]' "$SCHEMA_PATH" | sort
}

# Extract TokenType enum from C code
extract_c_tokens() {
    # Find the enum typedef TokenType
    awk '/typedef enum.*{/,/} TokenType;/' "$C_GENERATED" | \
        grep -E '^\s*(TOKEN_[A-Z_0-9]+)' | \
        sed -E 's/^[[:space:]]*//; s/[,=].*//; s,[[:space:]]*/\*.*\*/,,; s/[[:space:]]*$//' | \
        sort
}

# Extract LexerTokenType enum from NanoLang code
extract_nano_tokens() {
    awk '/enum LexerTokenType[[:space:]]*{/,/^}/' "$NANO_SCHEMA" | \
        grep -E '^\s*(TOKEN_[A-Z_0-9]+)' | \
        sed -E 's/^[[:space:]]*//; s/[,=].*//; s/#.*//; s/[[:space:]]*$//' | \
        sort
}

# Extract ParseNodeType enum from C code
extract_c_parse_nodes() {
    awk '/typedef enum.*{/,/} ParseNodeType;/' "$C_GENERATED" | \
        grep -E '^\s*(PNODE_[A-Z_0-9]+)' | \
        sed -E 's/^[[:space:]]*//; s/[,=].*//; s,[[:space:]]*/\*.*\*/,,; s/[[:space:]]*$//' | \
        sort
}

# Extract ParseNodeType enum from NanoLang code
extract_nano_parse_nodes() {
    awk '/enum ParseNodeType[[:space:]]*{/,/^}/' "$NANO_SCHEMA" | \
        grep -E '^\s*(PNODE_[A-Z_0-9]+)' | \
        sed -E 's/^[[:space:]]*//; s/[,=].*//; s/#.*//; s/[[:space:]]*$//' | \
        sort
}

# Compare two lists and report differences
compare_lists() {
    local name="$1"
    local schema_file="$2"
    local impl_file="$3"
    
    local missing=$(comm -23 "$schema_file" "$impl_file")
    local extra=$(comm -13 "$schema_file" "$impl_file")
    
    if [ -n "$missing" ]; then
        error "$name missing: $(echo $missing | tr '\n' ',' | sed 's/,$//')"
    fi
    
    if [ -n "$extra" ]; then
        warning "$name has extra: $(echo $extra | tr '\n' ',' | sed 's/,$//')"
    fi
    
    if [ -z "$missing" ] && [ -z "$extra" ]; then
        local count=$(wc -l < "$schema_file")
        passed "$name matches schema ($count items)"
    fi
}

# Main validation function
main() {
    echo "========================================================================"
    echo "SCHEMA VALIDATION"
    echo "========================================================================"
    echo ""
    
    # Check that files exist
    if [ ! -f "$SCHEMA_PATH" ]; then
        error "Schema file not found: $SCHEMA_PATH"
        exit 1
    fi
    
    if [ ! -f "$C_GENERATED" ]; then
        error "C generated file not found: $C_GENERATED"
        exit 1
    fi
    
    if [ ! -f "$NANO_SCHEMA" ]; then
        error "NanoLang schema file not found: $NANO_SCHEMA"
        exit 1
    fi
    
    passed "Found all required files"
    
    # Create temp directory for comparison
    TMPDIR=$(mktemp -d)
    trap "rm -rf $TMPDIR" EXIT
    
    # Check TokenType enums
    echo ""
    echo "Checking TokenType enums..."
    extract_schema_tokens > "$TMPDIR/schema_tokens"
    extract_c_tokens > "$TMPDIR/c_tokens"
    extract_nano_tokens > "$TMPDIR/nano_tokens"
    
    if [ ! -s "$TMPDIR/c_tokens" ]; then
        error "Could not extract TokenType enum from C generated code"
    else
        compare_lists "C TokenType" "$TMPDIR/schema_tokens" "$TMPDIR/c_tokens"
    fi
    
    if [ ! -s "$TMPDIR/nano_tokens" ]; then
        error "Could not extract LexerTokenType enum from NanoLang generated code"
    else
        compare_lists "NanoLang LexerTokenType" "$TMPDIR/schema_tokens" "$TMPDIR/nano_tokens"
    fi
    
    # Check ParseNodeType enums
    echo ""
    echo "Checking ParseNodeType enums..."
    extract_schema_parse_nodes > "$TMPDIR/schema_nodes"
    extract_c_parse_nodes > "$TMPDIR/c_nodes"
    extract_nano_parse_nodes > "$TMPDIR/nano_nodes"
    
    if [ ! -s "$TMPDIR/c_nodes" ]; then
        error "Could not extract ParseNodeType enum from C generated code"
    else
        compare_lists "C ParseNodeType" "$TMPDIR/schema_nodes" "$TMPDIR/c_nodes"
    fi
    
    if [ ! -s "$TMPDIR/nano_nodes" ]; then
        error "Could not extract ParseNodeType enum from NanoLang generated code"
    else
        compare_lists "NanoLang ParseNodeType" "$TMPDIR/schema_nodes" "$TMPDIR/nano_nodes"
    fi
    
    # Print summary
    echo ""
    echo "========================================================================"
    local total=$((ERRORS + WARNINGS + PASSED))
    local success_rate=0
    if [ $total -gt 0 ]; then
        success_rate=$((PASSED * 100 / total))
    fi
    echo "SUMMARY: $PASSED/$total checks passed ($success_rate%)"
    echo "Errors: $ERRORS, Warnings: $WARNINGS"
    echo "========================================================================"
    
    # Exit with error if there were any errors
    if [ $ERRORS -gt 0 ]; then
        exit 1
    fi
    
    exit 0
}

# Run main
main

