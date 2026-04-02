#!/usr/bin/env bash
# test_sourcemap.sh — Validate WASM source map emission (--target wasm)
#
# Tests that nanoc emits a valid .map sidecar JSON alongside the .wasm output
# when compiling with --target wasm.  Checks:
#   1. Source map file is created (<output>.map)
#   2. JSON parses (contains required top-level keys)
#   3. "version" is 1
#   4. "source" names the .nano file
#   5. "functions" array is non-empty
#   6. Each function entry has wasm_offset, src_line, src_col fields
#   7. wasm_offset values are non-negative integers

set +e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PASS=0
FAIL=0

pass() { echo "✅ $1"; PASS=$((PASS + 1)); }
fail() { echo "❌ $1"; FAIL=$((FAIL + 1)); }

# ── Create a test .nano file ───────────────────────────────────────────────
NANO_SRC="$(mktemp /tmp/test_sourcemap_XXXXXX.nano)"
WASM_OUT="$(mktemp /tmp/test_sourcemap_XXXXXX)"
MAP_OUT="${WASM_OUT}.map"

cat > "$NANO_SRC" << 'NANO_EOF'
fn add(a: int, b: int) -> int {
    return a + b
}
shadow add {
    assert (== (add 2 3) 5)
}
fn multiply(a: int, b: int) -> int {
    return a * b
}
shadow multiply {
    assert (== (multiply 3 4) 12)
}
fn main() -> int {
    return 0
}
shadow main { assert true }
NANO_EOF

# ── Compile to WASM ────────────────────────────────────────────────────────
COMPILE_OUT=$(bin/nanoc "$NANO_SRC" -o "$WASM_OUT" --target wasm 2>&1)
COMPILE_EXIT=$?

if [ $COMPILE_EXIT -ne 0 ]; then
    fail "nanoc --target wasm compilation failed: $COMPILE_OUT"
    rm -f "$NANO_SRC" "$WASM_OUT" "$MAP_OUT"
    echo ""
    echo "Results: $PASS passed, $FAIL failed"
    exit 1
fi

# ── Test 1: .wasm file was created ────────────────────────────────────────
if [ -f "$WASM_OUT" ] && [ -s "$WASM_OUT" ]; then
    pass ".wasm output file created and non-empty"
else
    fail ".wasm output file missing or empty"
fi

# ── Test 2: Source map file was created ──────────────────────────────────
if [ -f "$MAP_OUT" ] && [ -s "$MAP_OUT" ]; then
    pass "Source map file ($MAP_OUT) created and non-empty"
else
    fail "Source map file missing or empty"
    rm -f "$NANO_SRC" "$WASM_OUT" "$MAP_OUT"
    echo ""
    echo "Results: $PASS passed, $FAIL failed"
    exit 1
fi

# ── Parse the JSON with Python (always available) ────────────────────────
MAP_JSON=$(cat "$MAP_OUT")

# Test 3: version field = 1
VERSION=$(python3 -c "import json, sys; d=json.loads(sys.stdin.read()); print(d.get('version','MISSING'))" <<< "$MAP_JSON" 2>/dev/null)
if [ "$VERSION" = "1" ]; then
    pass "version field is 1"
else
    fail "version field: expected '1', got '$VERSION'"
fi

# Test 4: source field references the .nano filename
SOURCE=$(python3 -c "import json, sys; d=json.loads(sys.stdin.read()); print(d.get('source','MISSING'))" <<< "$MAP_JSON" 2>/dev/null)
NANO_BASENAME=$(basename "$NANO_SRC")
if echo "$SOURCE" | grep -q "\.nano"; then
    pass "source field references a .nano file ('$SOURCE')"
else
    fail "source field does not reference .nano file: '$SOURCE'"
fi

# Test 5: functions array is non-empty
FUNC_COUNT=$(python3 -c "import json, sys; d=json.loads(sys.stdin.read()); print(len(d.get('functions',[])))" <<< "$MAP_JSON" 2>/dev/null)
if [ "${FUNC_COUNT:-0}" -gt 0 ]; then
    pass "functions array has $FUNC_COUNT entries"
else
    fail "functions array is empty or missing"
fi

# Test 6: each function entry has required fields
FIELDS_OK=$(python3 - <<PYEOF
import json, sys
data = json.loads("""$MAP_JSON""")
fns = data.get('functions', [])
ok = True
for fn in fns:
    for field in ('name', 'wasm_offset', 'src_line', 'src_col'):
        if field not in fn:
            print(f"MISSING:{field} in {fn}")
            ok = False
            break
print("OK" if ok else "FAIL")
PYEOF
)
if [ "$FIELDS_OK" = "OK" ]; then
    pass "All function entries have name, wasm_offset, src_line, src_col fields"
else
    fail "Function entries missing required fields: $FIELDS_OK"
fi

# Test 7: wasm_offset values are non-negative integers
OFFSETS_OK=$(python3 - <<PYEOF
import json, sys
data = json.loads("""$MAP_JSON""")
fns = data.get('functions', [])
for fn in fns:
    off = fn.get('wasm_offset', -1)
    if not isinstance(off, int) or off < 0:
        print(f"BAD_OFFSET:{fn}")
        exit()
print("OK")
PYEOF
)
if [ "$OFFSETS_OK" = "OK" ]; then
    pass "All wasm_offset values are non-negative integers"
else
    fail "Invalid wasm_offset value: $OFFSETS_OK"
fi

# Test 8: WASM magic bytes (0x00 0x61 0x73 0x6d = '\0asm')
MAGIC=$(xxd -l 4 "$WASM_OUT" 2>/dev/null | awk '{print $2$3}' | head -1)
if [ "${MAGIC}" = "0061736d" ]; then
    pass "WASM file has correct magic bytes (\\0asm)"
else
    fail "WASM file missing magic bytes: $MAGIC"
fi

# ── Cleanup ────────────────────────────────────────────────────────────────
rm -f "$NANO_SRC" "$WASM_OUT" "$MAP_OUT"

echo ""
echo "Source map tests: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ] && exit 0 || exit 1
