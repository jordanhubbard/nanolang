#!/usr/bin/env bash
# tests/test_dwarf.sh — DWARF debug info emission tests
#
# Tests that --debug flag emits correct debug metadata for both LLVM IR
# and RISC-V assembly backends.
#
# Usage:
#   bash tests/test_dwarf.sh [compiler_path]
#   make test-dwarf

set -euo pipefail

COMPILER="${1:-./bin/nanoc_c}"
TMPDIR_TEST="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_TEST"' EXIT

PASS=0
FAIL=0

pass() { echo "  ✓ $1"; PASS=$((PASS + 1)); }
fail() { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }

# ── Create a simple test program ─────────────────────────────────────────────

cat > "$TMPDIR_TEST/foo.nano" <<'NANO'
fn add(a: int, b: int) -> int {
    let result = (+ a b)
    return result
}

fn main() -> int {
    let x = (add 1 2)
    return x
}
NANO

echo ""
echo "=========================================="
echo "DWARF Debug Info Emission Tests"
echo "=========================================="

# ── Test 1: LLVM IR with --debug ──────────────────────────────────────────────

echo ""
echo "Test 1: LLVM IR debug metadata (--llvm --debug)"

LL_OUT="$TMPDIR_TEST/foo.ll"
if "$COMPILER" "$TMPDIR_TEST/foo.nano" --llvm --debug -o "$LL_OUT" 2>/dev/null; then
    # Check for !DICompileUnit
    if grep -q "DICompileUnit" "$LL_OUT"; then
        pass "LLVM IR contains !DICompileUnit"
    else
        fail "LLVM IR missing !DICompileUnit"
        echo "    Output file: $LL_OUT"
    fi

    # Check for !DISubprogram
    if grep -q "DISubprogram" "$LL_OUT"; then
        pass "LLVM IR contains !DISubprogram"
    else
        fail "LLVM IR missing !DISubprogram"
    fi

    # Check for !DIFile
    if grep -q "DIFile" "$LL_OUT"; then
        pass "LLVM IR contains !DIFile"
    else
        fail "LLVM IR missing !DIFile"
    fi

    # Check that !dbg annotation is on function definitions
    if grep -q "!dbg" "$LL_OUT"; then
        pass "LLVM IR has !dbg annotations on functions"
    else
        fail "LLVM IR missing !dbg annotations"
    fi

    # Check that module-level metadata references are present
    if grep -q "!llvm.dbg.cu" "$LL_OUT"; then
        pass "LLVM IR has !llvm.dbg.cu module metadata"
    else
        fail "LLVM IR missing !llvm.dbg.cu"
    fi

    # Check DWARF version 4 flag
    if grep -q '"Dwarf Version"' "$LL_OUT" || grep -q 'Dwarf Version' "$LL_OUT"; then
        pass "LLVM IR specifies Dwarf Version"
    else
        fail "LLVM IR missing Dwarf Version flag"
    fi
else
    fail "Compiler failed on LLVM IR + debug mode"
fi

# ── Test 2: LLVM IR without --debug should NOT have debug metadata ──────────

echo ""
echo "Test 2: LLVM IR without --debug (no debug metadata)"

LL_NODEBUG="$TMPDIR_TEST/foo_nodebug.ll"
if "$COMPILER" "$TMPDIR_TEST/foo.nano" --llvm -o "$LL_NODEBUG" 2>/dev/null; then
    if grep -q "DICompileUnit" "$LL_NODEBUG"; then
        fail "Non-debug LLVM IR should not contain DICompileUnit"
    else
        pass "Non-debug LLVM IR has no debug metadata"
    fi
else
    fail "Compiler failed on LLVM IR (no debug)"
fi

# ── Test 3: RISC-V assembly with --debug ─────────────────────────────────────

echo ""
echo "Test 3: RISC-V assembly debug directives (--target riscv --debug)"

ASM_OUT="$TMPDIR_TEST/foo.s"
if "$COMPILER" "$TMPDIR_TEST/foo.nano" --target riscv --debug -o "$ASM_OUT" 2>/dev/null; then
    # Check for .file directive
    if grep -q "\.file" "$ASM_OUT"; then
        pass "RISC-V asm contains .file directive"
    else
        fail "RISC-V asm missing .file directive"
        echo "    Output file: $ASM_OUT"
    fi

    # Check for .loc directive
    if grep -q "\.loc" "$ASM_OUT"; then
        pass "RISC-V asm contains .loc directive"
    else
        fail "RISC-V asm missing .loc directive"
    fi

    # Check that .file references the source file
    if grep -q "foo\.nano" "$ASM_OUT"; then
        pass "RISC-V asm .file references the source file"
    else
        fail "RISC-V asm .file does not reference foo.nano"
    fi

    # Check for DWARF sections
    if grep -q "\.debug_abbrev\|\.debug_info" "$ASM_OUT"; then
        pass "RISC-V asm emits .debug_abbrev/.debug_info sections"
    else
        fail "RISC-V asm missing DWARF sections"
    fi
else
    fail "Compiler failed on RISC-V + debug mode"
fi

# ── Test 4: RISC-V without --debug should NOT have .file/.loc ────────────────

echo ""
echo "Test 4: RISC-V without --debug (no debug directives)"

ASM_NODEBUG="$TMPDIR_TEST/foo_nodebug.s"
if "$COMPILER" "$TMPDIR_TEST/foo.nano" --target riscv -o "$ASM_NODEBUG" 2>/dev/null; then
    if grep -q "\.loc" "$ASM_NODEBUG"; then
        fail "Non-debug RISC-V asm should not contain .loc directives"
    else
        pass "Non-debug RISC-V asm has no .loc directives"
    fi
else
    fail "Compiler failed on RISC-V (no debug)"
fi

# ── Summary ──────────────────────────────────────────────────────────────────

echo ""
echo "=========================================="
if [ $FAIL -eq 0 ]; then
    echo "DWARF tests PASSED: $PASS passed, 0 failed"
    echo "=========================================="
    exit 0
else
    echo "DWARF tests FAILED: $PASS passed, $FAIL failed"
    echo "=========================================="
    exit 1
fi
