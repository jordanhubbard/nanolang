#!/usr/bin/env bash
# tests/test_lsp_smoke.sh — basic smoke test for nanolang-lsp
#
# Sends initialize + textDocument/didOpen + textDocument/hover to the LSP
# server via stdin and checks the response contains expected JSON-RPC fields.

set -euo pipefail

LSP_BIN="${1:-bin/nanolang-lsp}"

if [ ! -x "$LSP_BIN" ]; then
    echo "SKIP: $LSP_BIN not found or not executable"
    exit 0
fi

# Helper: build a Content-Length framed LSP message
make_message() {
    local body="$1"
    local len=${#body}
    printf "Content-Length: %d\r\n\r\n%s" "$len" "$body"
}

# Create a temp nano file
TMP_DIR=$(mktemp -d)
TMP_FILE="$TMP_DIR/test.nano"
cat > "$TMP_FILE" <<'EOF'
fn add(a: int, b: int) -> int {
    let result: int = a + b;
    result
}

fn main() -> void {
    let x: int = add(1, 2);
    println x;
}
EOF

URI="file://$TMP_FILE"

# Build all LSP messages to send
INIT=$(make_message '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}')
INITIALIZED=$(make_message '{"jsonrpc":"2.0","method":"initialized","params":{}}')
DID_OPEN=$(make_message "{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/didOpen\",\"params\":{\"textDocument\":{\"uri\":\"$URI\",\"languageId\":\"nanolang\",\"version\":1,\"text\":\"fn add(a: int, b: int) -> int {\\n    let result: int = a + b;\\n    result\\n}\\n\\nfn main() -> void {\\n    let x: int = add(1, 2);\\n    println x;\\n}\\n\"}}}")
HOVER=$(make_message "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"textDocument/hover\",\"params\":{\"textDocument\":{\"uri\":\"$URI\"},\"position\":{\"line\":1,\"character\":8}}}")
SHUTDOWN=$(make_message '{"jsonrpc":"2.0","id":3,"method":"shutdown","params":{}}')
EXIT_MSG=$(make_message '{"jsonrpc":"2.0","method":"exit","params":{}}')

# Run LSP server with a timeout, feed all messages
RESPONSE=$(printf '%s%s%s%s%s%s' \
    "$INIT" "$INITIALIZED" "$DID_OPEN" "$HOVER" "$SHUTDOWN" "$EXIT_MSG" \
    | timeout 10 "$LSP_BIN" 2>/dev/null || true)

# Cleanup
rm -rf "$TMP_DIR"

# Check responses

# 1. initialize response should contain "capabilities"
if echo "$RESPONSE" | grep -q '"capabilities"'; then
    echo "PASS: initialize response contains capabilities"
else
    echo "FAIL: initialize response missing capabilities"
    echo "Got: $RESPONSE"
    exit 1
fi

# 2. publishDiagnostics notification should be sent after didOpen
if echo "$RESPONSE" | grep -q '"textDocument/publishDiagnostics"'; then
    echo "PASS: publishDiagnostics notification received"
else
    echo "FAIL: publishDiagnostics not received"
    echo "Got: $RESPONSE"
    exit 1
fi

# 3. hover response (id:2) should be a valid JSON-RPC result
if echo "$RESPONSE" | grep -q '"id":2'; then
    echo "PASS: hover response received"
else
    echo "FAIL: hover response not received"
    echo "Got: $RESPONSE"
    exit 1
fi

# 4. shutdown response (id:3) should be present
if echo "$RESPONSE" | grep -q '"id":3'; then
    echo "PASS: shutdown response received"
else
    echo "FAIL: shutdown response not received"
    echo "Got: $RESPONSE"
    exit 1
fi

echo ""
echo "All LSP smoke tests passed."
