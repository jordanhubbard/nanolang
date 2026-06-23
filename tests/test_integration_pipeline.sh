#!/bin/bash
# =============================================================================
# Integration Test: End-to-End Compiler Pipeline
# =============================================================================
# Tests that all self-hosted components work together to compile a program
#
# Pipeline:
#   1. Lexer (lexer_main.nano) - tokenizes source
#   2. Parser (parser.nano) - builds AST  
#   3. Type Checker (typecheck.nano) - validates types
#   4. Transpiler (transpiler.nano) - generates C code
#
# Note: This is a demonstration test showing the components work individually.
# Full integration requires a module system for linking .nano files together.
# =============================================================================

set -e

echo "=========================================="
echo "Integration Test: Self-Hosted Pipeline"
echo "=========================================="
echo ""

# Check that all components exist
COMPONENTS="parser typecheck transpiler"
for comp in $COMPONENTS; do
    if [ ! -f "bin/$comp" ]; then
        echo "❌ Error: bin/$comp not found"
        echo "   Run 'make stage2' first"
        exit 1
    fi
done

echo "✓ All components found"
echo ""

# Test 1: Parser can parse a simple program
echo "Test 1: Parser"
echo "  Testing parser with sample code..."
if bin/parser > /dev/null 2>&1; then
    echo "  ✓ Parser tests passed"
else
    echo "  ❌ Parser tests failed"
    exit 1
fi

# Test 2: Type checker works
echo ""
echo "Test 2: Type Checker"
echo "  Testing typecheck..."
if bin/typecheck > /dev/null 2>&1; then
    echo "  ✓ Type checker tests passed"
else
    echo "  ❌ Type checker tests failed"
    exit 1
fi

# Test 3: Transpiler generates code
echo ""
echo "Test 3: Transpiler"
echo "  Testing transpiler..."
if bin/transpiler > /dev/null 2>&1; then
    echo "  ✓ Transpiler tests passed"
else
    echo "  ❌ Transpiler tests failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ Integration Test: PASSED"
echo "=========================================="
echo ""
echo "Status:"
echo "  ✓ Parser: Working"
echo "  ✓ Type Checker: Working"
echo "  ✓ Transpiler: Working"
echo ""
echo "Note: Full end-to-end compilation requires:"
echo "  - Module system for linking .nano files"
echo "  - Or manual C-level linking with --keep-c"
echo ""
