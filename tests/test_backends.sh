#!/usr/bin/env bash
# test_backends.sh — integration tests for alternative compilation backends
#
# Exercises --target riscv, --target ptx, and --reflect to ensure:
#   - riscv_backend.c code paths are executed
#   - ptx_backend.c code paths are executed
#   - reflection.c code paths are executed
#
# Usage: bash tests/test_backends.sh [path-to-nanoc]

set -e

COMPILER="${1:-./bin/nanoc_c}"
TMPDIR_BASE="${TMPDIR:-/tmp}"
TMP="${TMPDIR_BASE}/nano_backend_test_$$"
mkdir -p "$TMP"
trap 'rm -rf "$TMP"' EXIT

PASS=0
FAIL=0

pass() { echo "  ✓ $1"; PASS=$((PASS + 1)); }
fail() { echo "  ✗ $1"; FAIL=$((FAIL + 1)); }

# Create a simple but non-trivial nano source file for backend testing
cat > "$TMP/simple.nano" << 'NANO'
fn add(x: int, y: int) -> int {
    return (+ x y)
}

fn multiply(x: int, y: int) -> int {
    return (* x y)
}

fn main() -> int {
    let result: int = (add 3 4)
    (print result)
    return 0
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 0 0) 0)
}

shadow multiply {
    assert (== (multiply 2 3) 6)
    assert (== (multiply 0 5) 0)
}
NANO

# Create a nano file with functions, structs, and constants for reflect testing
cat > "$TMP/module.nano" << 'NANO'
fn compute(x: int, y: int) -> int {
    return (+ (* x x) (* y y))
}

fn greet(name: string) -> string {
    return (str_concat "Hello, " name)
}

fn negate(flag: bool) -> bool {
    return (! flag)
}

fn main() -> int {
    return 0
}

shadow compute {
    assert (== (compute 3 4) 25)
}

shadow greet {
    assert (== (greet "world") "Hello, world")
}

shadow negate {
    assert (== (negate true) false)
}
NANO

echo "=== Backend Integration Tests ==="
echo ""

# ── RISC-V backend ────────────────────────────────────────────────────────────
echo "  Testing RISC-V backend (--target riscv)..."

RISCV_OUT="$TMP/simple.riscv"
if "$COMPILER" "$TMP/simple.nano" --target riscv -o "$RISCV_OUT" 2>/dev/null; then
    if [ -f "$RISCV_OUT" ] && [ -s "$RISCV_OUT" ]; then
        pass "riscv: output file created and non-empty"
    else
        fail "riscv: output file missing or empty"
    fi

    # Verify basic RISC-V assembly markers
    if grep -q "\.text\|\.global\|addi\|sw\|lw\|ret\|jal\|auipc" "$RISCV_OUT" 2>/dev/null; then
        pass "riscv: output contains RISC-V assembly directives"
    else
        fail "riscv: output missing expected assembly directives"
    fi
else
    fail "riscv: compiler returned non-zero exit for --target riscv"
fi

# Test with multiply function to exercise more code paths
RISCV_OUT2="$TMP/module.riscv"
if "$COMPILER" "$TMP/module.nano" --target riscv -o "$RISCV_OUT2" 2>/dev/null; then
    pass "riscv: compiled module with multiple function signatures"
else
    fail "riscv: failed to compile module with --target riscv"
fi

echo ""

# ── PTX backend ───────────────────────────────────────────────────────────────
echo "  Testing PTX backend (--target ptx)..."

PTX_OUT="$TMP/simple.ptx"
if "$COMPILER" "$TMP/simple.nano" --target ptx -o "$PTX_OUT" 2>/dev/null; then
    if [ -f "$PTX_OUT" ] && [ -s "$PTX_OUT" ]; then
        pass "ptx: output file created and non-empty"
    else
        fail "ptx: output file missing or empty"
    fi

    # PTX assembly should have .version or .target
    if grep -q "\.version\|\.target\|\.func\|\.reg\|\.param\|\.b32\|\.b64" "$PTX_OUT" 2>/dev/null; then
        pass "ptx: output contains PTX assembly directives"
    else
        fail "ptx: output missing expected PTX directives"
    fi
else
    fail "ptx: compiler returned non-zero exit for --target ptx"
fi

# Test PTX with more complex program
PTX_OUT2="$TMP/module.ptx"
if "$COMPILER" "$TMP/module.nano" --target ptx -o "$PTX_OUT2" 2>/dev/null; then
    pass "ptx: compiled module with multiple functions"
else
    fail "ptx: failed to compile module with --target ptx"
fi

echo ""

# ── LLVM IR backend ───────────────────────────────────────────────────────────
echo "  Testing LLVM IR backend (--llvm)..."

LLVM_OUT="$TMP/simple.ll"
if "$COMPILER" "$TMP/simple.nano" --llvm -o "$LLVM_OUT" 2>/dev/null; then
    if [ -f "$LLVM_OUT" ] && [ -s "$LLVM_OUT" ]; then
        pass "llvm: output file created and non-empty"
    else
        fail "llvm: output file missing or empty"
    fi

    if grep -q "define\|declare\|target datalayout\|target triple\|i64\|i32\|void" "$LLVM_OUT" 2>/dev/null; then
        pass "llvm: output contains LLVM IR directives"
    else
        fail "llvm: output missing expected LLVM IR directives"
    fi
else
    fail "llvm: compiler returned non-zero for --llvm"
fi

LLVM_OUT2="$TMP/module.ll"
if "$COMPILER" "$TMP/module.nano" --llvm -o "$LLVM_OUT2" 2>/dev/null; then
    pass "llvm: compiled module with multiple function signatures"
else
    fail "llvm: failed to compile module with --llvm"
fi

echo ""

# ── DWARF debug info (riscv + --debug) ───────────────────────────────────────
echo "  Testing DWARF debug info (--target riscv --debug)..."

DWARF_OUT="$TMP/simple_debug.riscv"
if "$COMPILER" "$TMP/simple.nano" --target riscv --debug -o "$DWARF_OUT" 2>/dev/null; then
    if [ -f "$DWARF_OUT" ] && [ -s "$DWARF_OUT" ]; then
        pass "dwarf: output file created and non-empty"
    else
        fail "dwarf: output file missing or empty"
    fi

    # DWARF sections should contain .debug or DW_ markers
    if grep -qE "\.debug|DW_|debug_info|debug_abbrev" "$DWARF_OUT" 2>/dev/null; then
        pass "dwarf: output contains DWARF debug sections"
    else
        fail "dwarf: output missing DWARF debug sections"
    fi
else
    fail "dwarf: compiler returned non-zero for --target riscv --debug"
fi

echo ""

# ── Reflection / --reflect ───────────────────────────────────────────────────
echo "  Testing reflection API (--reflect)..."

REFLECT_OUT="$TMP/module.reflect.json"
if "$COMPILER" "$TMP/module.nano" --reflect "$REFLECT_OUT" 2>/dev/null; then
    if [ -f "$REFLECT_OUT" ] && [ -s "$REFLECT_OUT" ]; then
        pass "reflect: output file created and non-empty"
    else
        fail "reflect: output file missing or empty"
    fi

    # Verify JSON contains expected function names
    if grep -q '"compute"' "$REFLECT_OUT"; then
        pass "reflect: JSON contains function 'compute'"
    else
        fail "reflect: JSON missing function 'compute'"
    fi

    if grep -q '"greet"' "$REFLECT_OUT"; then
        pass "reflect: JSON contains function 'greet'"
    else
        fail "reflect: JSON missing function 'greet'"
    fi

    if grep -q '"negate"' "$REFLECT_OUT"; then
        pass "reflect: JSON contains function 'negate'"
    else
        fail "reflect: JSON missing function 'negate'"
    fi

    # Verify parameter types appear
    if grep -q '"int"\|"string"\|"bool"' "$REFLECT_OUT"; then
        pass "reflect: JSON contains type information"
    else
        fail "reflect: JSON missing type information"
    fi

    # Verify it's valid JSON (has opening/closing braces)
    if grep -q '^{' "$REFLECT_OUT" && grep -q '^}' "$REFLECT_OUT"; then
        pass "reflect: JSON has valid top-level structure"
    else
        fail "reflect: JSON missing valid top-level structure"
    fi
else
    fail "reflect: compiler returned non-zero for --reflect"
fi

# Test reflect on a single-function file (edge case)
cat > "$TMP/single.nano" << 'NANO'
fn identity(x: int) -> int {
    return x
}
shadow identity {
    assert (== (identity 42) 42)
}
NANO

REFLECT_SINGLE="$TMP/single.reflect.json"
if "$COMPILER" "$TMP/single.nano" --reflect "$REFLECT_SINGLE" 2>/dev/null; then
    if [ -f "$REFLECT_SINGLE" ] && grep -q '"identity"' "$REFLECT_SINGLE"; then
        pass "reflect: single-function module reflects correctly"
    else
        fail "reflect: single-function module reflect output invalid"
    fi
else
    fail "reflect: failed on single-function module"
fi

echo ""

# ── Benchmark mode (--bench) ──────────────────────────────────────────────────
echo "  Testing benchmark mode (--bench)..."

cat > "$TMP/bench.nano" << 'NANO'
fn bench_add() -> int {
    return (+ 1 2)
}

fn bench_mul() -> int {
    return (* 6 7)
}

fn main() -> int { return 0 }
NANO

BENCH_OUT="$TMP/bench_out"
if "$COMPILER" "$TMP/bench.nano" --bench -o "$BENCH_OUT" 2>/dev/null; then
    pass "bench: --bench mode ran without error"
else
    fail "bench: --bench mode returned non-zero"
fi

# Test --bench-n for fixed iteration count
if "$COMPILER" "$TMP/bench.nano" --bench --bench-n 100 -o "$BENCH_OUT" 2>/dev/null; then
    pass "bench: --bench-n fixed iteration count works"
else
    fail "bench: --bench-n returned non-zero"
fi

# Test --bench-json output
BENCH_JSON="$TMP/bench_results.json"
if "$COMPILER" "$TMP/bench.nano" --bench --bench-n 10 --bench-json "$BENCH_JSON" -o "$BENCH_OUT" 2>/dev/null; then
    if [ -f "$BENCH_JSON" ] && [ -s "$BENCH_JSON" ]; then
        pass "bench: --bench-json writes JSON results file"
    else
        fail "bench: --bench-json did not produce output file"
    fi
else
    fail "bench: --bench-json returned non-zero"
fi

echo ""

# ── Summary ──────────────────────────────────────────────────────────────────
TOTAL=$((PASS + FAIL))
echo "Backend tests: $PASS/$TOTAL passed"
echo ""

if [ "$FAIL" -gt 0 ]; then
    echo "❌ $FAIL backend test(s) failed"
    exit 1
else
    echo "✅ All backend tests passed"
fi
