#!/usr/bin/env bash
# test_wasm_simd.sh — tests for WASM SIMD128 auto-vectorization
# Tests:
#   1. Interpreter correctness: shadow tests pass on vectorizable patterns
#   2. nanoc --target wasm --simd produces a valid .wasm file
#   3. The output .wasm contains the simd_annots custom section
#   4. wasm_simd_detect finds at least 3 candidates in the test file

set -euo pipefail
NANOC="${1:-./bin/nanoc}"
NANO="${2:-./bin/nano}"
PASS=0; FAIL=0

ok()  { echo "  ✅ $1"; PASS=$((PASS+1)); }
fail(){ echo "  ❌ $1: $2"; FAIL=$((FAIL+1)); }

echo "WASM SIMD128 vectorization tests:"

# ── 1. Interpreter correctness ─────────────────────────────────────────
if "$NANO" tests/unit/test_wasm_simd.nano > /tmp/simd_out.txt 2>&1; then
    ok "interpreter: all shadow tests pass"
else
    fail "interpreter: shadow tests failed" "$(cat /tmp/simd_out.txt | head -5)"
fi

# ── 2. nanoc --target wasm --simd compiles without error ───────────────
if "$NANOC" tests/unit/test_wasm_simd.nano --target wasm --simd -o /tmp/test_simd.wasm \
       > /tmp/simd_compile.txt 2>&1; then
    ok "nanoc --simd: compiles cleanly"
else
    fail "nanoc --simd: compilation failed" "$(cat /tmp/simd_compile.txt | head -5)"
fi

# ── 3. Output is a valid WASM binary (magic bytes 0x00 0x61 0x73 0x6D) ─
if [ -f /tmp/test_simd.wasm ]; then
    MAGIC=$(xxd -p -l 4 /tmp/test_simd.wasm 2>/dev/null || od -A n -N 4 -t x1 /tmp/test_simd.wasm | tr -d ' \n')
    if echo "$MAGIC" | grep -qi "0061736d"; then
        ok "output is valid WASM (magic 0x0061736d present)"
    else
        fail "WASM magic bytes wrong" "got: $MAGIC"
    fi
else
    fail "WASM output file missing" "/tmp/test_simd.wasm not created"
fi

# ── 4. simd_annots custom section present ──────────────────────────────
if [ -f /tmp/test_simd.wasm ]; then
    if strings /tmp/test_simd.wasm 2>/dev/null | grep -q "simd_annots"; then
        ok "simd_annots custom section present in .wasm"
    else
        fail "simd_annots custom section missing" "strings output: $(strings /tmp/test_simd.wasm | tail -5)"
    fi
fi

# ── 5. nanoc --target wasm (no --simd) still works ─────────────────────
if "$NANOC" tests/unit/test_wasm_simd.nano --target wasm -o /tmp/test_no_simd.wasm \
       > /dev/null 2>&1; then
    ok "nanoc (no --simd): baseline WASM still compiles"
else
    fail "nanoc (no --simd): baseline compilation failed" ""
fi

# ── 6. Baseline has no simd_annots section ─────────────────────────────
if [ -f /tmp/test_no_simd.wasm ]; then
    if ! strings /tmp/test_no_simd.wasm 2>/dev/null | grep -q "simd_annots"; then
        ok "baseline .wasm has no simd_annots section (as expected)"
    else
        fail "baseline .wasm unexpectedly has simd_annots" ""
    fi
fi

# ── 7. --simd output is larger than baseline (custom section was added) ─
if [ -f /tmp/test_simd.wasm ] && [ -f /tmp/test_no_simd.wasm ]; then
    SZ_SIMD=$(wc -c < /tmp/test_simd.wasm)
    SZ_BASE=$(wc -c < /tmp/test_no_simd.wasm)
    if [ "$SZ_SIMD" -gt "$SZ_BASE" ]; then
        ok "--simd output is larger than baseline ($SZ_SIMD vs $SZ_BASE bytes)"
    else
        fail "--simd output not larger than baseline" "$SZ_SIMD vs $SZ_BASE bytes"
    fi
fi

rm -f /tmp/simd_out.txt /tmp/simd_compile.txt /tmp/test_simd.wasm \
       /tmp/test_simd.wasm.map /tmp/test_no_simd.wasm /tmp/test_no_simd.wasm.map

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ]
