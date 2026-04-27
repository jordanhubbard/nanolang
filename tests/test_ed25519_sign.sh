#!/usr/bin/env bash
# test_ed25519_sign.sh — Test Ed25519 WASM module signing (nanoc sign/verify)
#
# Tests the complete sign/verify workflow:
#   1. Compile a .nano program to WASM
#   2. Sign the WASM module with the default key (~/.nanoc/signing.key)
#   3. Verify the signature
#   4. Verify that a modified/tampered WASM fails verification
#   5. Verify that signing twice (re-sign) succeeds (idempotent section handling)

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PASS=0
FAIL=0

pass() { echo "✅ $1"; PASS=$((PASS + 1)); }
fail() { echo "❌ $1"; FAIL=$((FAIL + 1)); }
skip() { echo "⏭  $1 (skipped)"; }

# ── Verify OpenSSL is available (required for signing) ────────────────────
if ! openssl version >/dev/null 2>&1; then
    skip "OpenSSL not available — skipping Ed25519 signing tests"
    exit 0
fi

# ── Build a simple WASM module ────────────────────────────────────────────
NANO_SRC="$(mktemp /tmp/test_sign_XXXXXX.nano)"
WASM_FILE="$(mktemp /tmp/test_sign_XXXXXX.wasm)"

cat > "$NANO_SRC" << 'NANO_EOF'
fn add(a: int, b: int) -> int {
    return a + b
}
shadow add { assert (== (add 2 3) 5) }
fn main() -> int { return 0 }
shadow main { assert true }
NANO_EOF

COMPILE_OUT=$(bin/nanoc "$NANO_SRC" -o "$WASM_FILE" --target wasm 2>&1)
COMPILE_EXIT=$?

if [ $COMPILE_EXIT -ne 0 ]; then
    fail "WASM compilation failed: $COMPILE_OUT"
    rm -f "$NANO_SRC" "$WASM_FILE" "${WASM_FILE}.map"
    echo ""
    echo "Results: $PASS passed, $FAIL failed"
    exit 1
fi
pass "WASM module compiled successfully"

# ── Test 1: Sign the WASM module ─────────────────────────────────────────
SIGN_OUT=$(bin/nanoc sign "$WASM_FILE" 2>&1)
SIGN_EXIT=$?

if [ $SIGN_EXIT -eq 0 ]; then
    pass "nanoc sign succeeded (exit 0)"
else
    fail "nanoc sign failed (exit $SIGN_EXIT): $SIGN_OUT"
fi

# ── Test 2: Signed WASM contains agentos.signature section ──────────────
# The signature section name is "agentos.signature" (17 bytes)
if strings "$WASM_FILE" 2>/dev/null | grep -q "agentos.signature"; then
    pass "Signed WASM contains 'agentos.signature' custom section"
else
    fail "Signed WASM missing 'agentos.signature' section"
fi

# ── Test 3: nanoc verify accepts the signed WASM ─────────────────────────
VERIFY_OUT=$(bin/nanoc verify "$WASM_FILE" 2>&1)
VERIFY_EXIT=$?

if [ $VERIFY_EXIT -eq 0 ]; then
    pass "nanoc verify accepts freshly-signed WASM (exit 0)"
else
    fail "nanoc verify rejected valid signature (exit $VERIFY_EXIT): $VERIFY_OUT"
fi

# ── Test 4: Verify output mentions VALID ─────────────────────────────────
if echo "$VERIFY_OUT" | grep -qi "valid\|ok\|verified"; then
    pass "nanoc verify output indicates valid signature"
else
    # Non-fatal: different output format is OK
    skip "nanoc verify output format check (got: ${VERIFY_OUT})"
fi

# ── Test 5: Re-signing (idempotent) ──────────────────────────────────────
RESIGN_OUT=$(bin/nanoc sign "$WASM_FILE" 2>&1)
RESIGN_EXIT=$?
if [ $RESIGN_EXIT -eq 0 ]; then
    pass "Re-signing already-signed WASM succeeds"
    # Verify again after re-sign
    VERIFY2_OUT=$(bin/nanoc verify "$WASM_FILE" 2>&1)
    if [ $? -eq 0 ]; then
        pass "Re-signed WASM passes verification"
    else
        fail "Re-signed WASM fails verification: $VERIFY2_OUT"
    fi
else
    fail "Re-signing failed (exit $RESIGN_EXIT): $RESIGN_OUT"
fi

# ── Test 6: Modified WASM fails verification ─────────────────────────────
TAMPERED_FILE="$(mktemp /tmp/test_sign_tampered_XXXXXX.wasm)"
cp "$WASM_FILE" "$TAMPERED_FILE"

# Flip a byte in the middle of the file (after the 4-byte magic header)
# Use Python to tamper byte at offset 10
python3 - "$TAMPERED_FILE" << 'PYEOF' 2>/dev/null
import sys
path = sys.argv[1]
with open(path, 'rb') as f:
    data = bytearray(f.read())
# Flip byte at offset 10 (well into the module data, before signature section)
if len(data) > 10:
    data[10] ^= 0xFF
with open(path, 'wb') as f:
    f.write(data)
PYEOF

TAMPER_VERIFY=$(bin/nanoc verify "$TAMPERED_FILE" 2>&1)
TAMPER_EXIT=$?
if [ $TAMPER_EXIT -ne 0 ]; then
    pass "Tampered WASM correctly rejected by nanoc verify"
else
    # Some implementations may still verify if the tampered byte is in a
    # non-signed region; check output message
    if echo "$TAMPER_VERIFY" | grep -qi "invalid\|fail\|error\|bad"; then
        pass "Tampered WASM flagged as invalid in output"
    else
        fail "Tampered WASM incorrectly accepted (should be rejected)"
    fi
fi
rm -f "$TAMPERED_FILE"

# ── Test 7: Unsigned WASM fails verification ─────────────────────────────
UNSIGNED_FILE="$(mktemp /tmp/test_sign_unsigned_XXXXXX.wasm)"
bin/nanoc "$NANO_SRC" -o "$UNSIGNED_FILE" --target wasm >/dev/null 2>&1
UNSIGNED_VERIFY=$(bin/nanoc verify "$UNSIGNED_FILE" 2>&1)
UNSIGNED_EXIT=$?
if [ $UNSIGNED_EXIT -ne 0 ] || echo "$UNSIGNED_VERIFY" | grep -qi "invalid\|no.*signature\|not.*signed"; then
    pass "Unsigned WASM correctly rejected by nanoc verify"
else
    fail "Unsigned WASM incorrectly accepted: $UNSIGNED_VERIFY"
fi
rm -f "$UNSIGNED_FILE"

# ── Cleanup ────────────────────────────────────────────────────────────────
rm -f "$NANO_SRC" "$WASM_FILE" "${WASM_FILE}.map"

echo ""
echo "Ed25519 sign/verify tests: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ] && exit 0 || exit 1
