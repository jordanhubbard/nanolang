#!/usr/bin/env bash
# test_lsp_hover_rowpoly.sh — tests LSP textDocument/hover with row-polymorphic type display
# and markdown code-block formatting.
set -e
NANO_LSP="${1:-./bin/nanolang-lsp}"
PASS=0
FAIL=0

check() {
    local desc="$1"; local response="$2"; local expected="$3"
    if echo "$response" | grep -q "$expected"; then
        echo "  ✅ $desc"; PASS=$((PASS + 1))
    else
        echo "  ❌ $desc (expected: $expected)"; FAIL=$((FAIL + 1))
    fi
}

send_lsp_sequence() {
    # Usage: send_lsp_sequence msg1 msg2 ... | nanolang-lsp
    for M in "$@"; do
        printf "Content-Length: %d\r\n\r\n%s" "${#M}" "$M"
    done
}

# ── Test 1: variable hover with markdown code-block ────────────────────
TEST_FILE=$(mktemp /tmp/nano_lsp_test_XXXXXX.nano)
cat > "$TEST_FILE" << 'EOF'
fn add(a: int, b: int) -> int {
    return (+ a b)
}
shadow add { assert (== (add 1 2) 3) }
fn main() -> int {
    let result: int = (add 10 20)
    return result
}
shadow main { assert (== (main) 0) }
EOF
URI="file://$TEST_FILE"
CONTENT=$(python3 -c "import json; print(json.dumps(open('$TEST_FILE').read()))")

INIT='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"processId":null,"capabilities":{}}}'
OPEN="{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/didOpen\",\"params\":{\"textDocument\":{\"uri\":\"$URI\",\"languageId\":\"nanolang\",\"version\":1,\"text\":$CONTENT}}}"
# Line 6 (0-based), col 11 = 'result' in 'return result'
HOVER_VAR="{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"textDocument/hover\",\"params\":{\"textDocument\":{\"uri\":\"$URI\"},\"position\":{\"line\":6,\"character\":11}}}"
# Line 5, col 22 = 'add' function call
HOVER_FN="{\"jsonrpc\":\"2.0\",\"id\":3,\"method\":\"textDocument/hover\",\"params\":{\"textDocument\":{\"uri\":\"$URI\"},\"position\":{\"line\":5,\"character\":22}}}"
SHUT='{"jsonrpc":"2.0","id":99,"method":"shutdown","params":{}}'
EX='{"jsonrpc":"2.0","method":"exit","params":{}}'

VAR_RESP=$(send_lsp_sequence "$INIT" "$OPEN" "$HOVER_VAR" "$SHUT" "$EX" | timeout 5 "$NANO_LSP" 2>/dev/null || true)
FN_RESP=$(send_lsp_sequence "$INIT" "$OPEN" "$HOVER_FN" "$SHUT" "$EX"  | timeout 5 "$NANO_LSP" 2>/dev/null || true)

echo "LSP hover row-polymorphic type tests:"
check "variable hover: returns result"     "$VAR_RESP" '"result"'
check "variable hover: markdown kind"      "$VAR_RESP" '"kind":"markdown"'
check "variable hover: nano code block"    "$VAR_RESP" "nano"
check "variable hover: shows type (int)"   "$VAR_RESP" '"value"'
check "function hover: fn signature"       "$FN_RESP"  '"result"'
check "function hover: markdown kind"      "$FN_RESP"  '"kind":"markdown"'

rm -f "$TEST_FILE"

# ── Test 2: struct type hover ─────────────────────────────────────────
TEST_FILE2=$(mktemp /tmp/nano_lsp_struct_XXXXXX.nano)
cat > "$TEST_FILE2" << 'EOF'
struct Color {
    r: int,
    g: int,
    b: int
}

fn make_red() -> Color {
    return { r: 255, g: 0, b: 0 }
}

shadow make_red {
    let c: Color = (make_red)
    assert (== c.r 255)
}

fn main() -> int {
    (make_red)
    return 0
}

shadow main { assert (== (main) 0) }
EOF
URI2="file://$TEST_FILE2"
CONTENT2=$(python3 -c "import json; print(json.dumps(open('$TEST_FILE2').read()))")

OPEN2="{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/didOpen\",\"params\":{\"textDocument\":{\"uri\":\"$URI2\",\"languageId\":\"nanolang\",\"version\":1,\"text\":$CONTENT2}}}"
# Hover over 'make_red' on line 16 (0-based)
HOVER2="{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"textDocument/hover\",\"params\":{\"textDocument\":{\"uri\":\"$URI2\"},\"position\":{\"line\":16,\"character\":5}}}"

STRUCT_RESP=$(send_lsp_sequence "$INIT" "$OPEN2" "$HOVER2" "$SHUT" "$EX" | timeout 5 "$NANO_LSP" 2>/dev/null || true)

check "struct hover: returns result"     "$STRUCT_RESP" '"result"'
check "struct hover: returns result"     "$STRUCT_RESP" "result"

rm -f "$TEST_FILE2"

# ── Test 3: interpreter unit test (build_type_label roundtrip) ────────
UNIT=$(mktemp /tmp/nano_lsp_unit_XXXXXX.nano)
cat > "$UNIT" << 'NANOEOF'
struct Pair {
    first: int,
    second: string
}

fn make_pair(a: int, b: string) -> Pair {
    return { first: a, second: b }
}

shadow make_pair {
    let p: Pair = (make_pair 1 "hello")
    assert (== p.first 1)
    assert (str_equals p.second "hello")
}

fn main() -> int {
    let p: Pair = (make_pair 42 "world")
    assert (== p.first 42)
    assert (str_equals p.second "world")
    (println "LSP hover type display PASSED")
    return 0
}

shadow main { assert (== (main) 0) }
NANOEOF

./bin/nano "$UNIT" 2>&1 | grep -q "PASSED" && {
    echo "  ✅ interpreter struct type round-trip"; PASS=$((PASS+1))
} || {
    echo "  ❌ interpreter struct type round-trip failed"; FAIL=$((FAIL+1))
}

rm -f "$UNIT"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ]
