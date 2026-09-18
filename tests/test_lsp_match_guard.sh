#!/usr/bin/env bash
# I require hover lookup to retain identifiers that appear only in a match guard.

set -euo pipefail

LSP_BIN="${1:-bin/nanolang-lsp}"
if [ ! -x "$LSP_BIN" ]; then
    echo "FAIL: $LSP_BIN is not executable"
    exit 1
fi

make_message() {
    local body="$1"
    printf 'Content-Length: %d\r\n\r\n%s' "${#body}" "$body"
}

TMP_DIR=$(mktemp -d)
trap 'rm -rf "$TMP_DIR"' EXIT
TMP_FILE="$TMP_DIR/match_guard.nano"
URI="file://$TMP_FILE"

SOURCE='union Choice { Some { value: int }, None {} }
fn ready(value: int) -> bool { return (> value 0) }
fn main() -> int {
    let choice: Choice = Choice.Some { value: 7 }
    return match choice {
        Some(payload) if (ready payload.value) => payload.value,
        _ => 0
    }
}
'
printf '%s' "$SOURCE" > "$TMP_FILE"

INIT=$(make_message '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"capabilities":{}}}')
INITIALIZED=$(make_message '{"jsonrpc":"2.0","method":"initialized","params":{}}')
DID_OPEN=$(make_message "{\"jsonrpc\":\"2.0\",\"method\":\"textDocument/didOpen\",\"params\":{\"textDocument\":{\"uri\":\"$URI\",\"languageId\":\"nanolang\",\"version\":1,\"text\":\"union Choice { Some { value: int }, None {} }\\nfn ready(value: int) -> bool { return (> value 0) }\\nfn main() -> int {\\n    let choice: Choice = Choice.Some { value: 7 }\\n    return match choice {\\n        Some(payload) if (ready payload.value) => payload.value,\\n        _ => 0\\n    }\\n}\\n\"}}}")
HOVER=$(make_message "{\"jsonrpc\":\"2.0\",\"id\":2,\"method\":\"textDocument/hover\",\"params\":{\"textDocument\":{\"uri\":\"$URI\"},\"position\":{\"line\":5,\"character\":28}}}")
SHUTDOWN=$(make_message '{"jsonrpc":"2.0","id":3,"method":"shutdown","params":{}}')
EXIT_MSG=$(make_message '{"jsonrpc":"2.0","method":"exit","params":{}}')

RESPONSE=$(printf '%s%s%s%s%s%s' \
    "$INIT" "$INITIALIZED" "$DID_OPEN" "$HOVER" "$SHUTDOWN" "$EXIT_MSG" \
    | timeout 10 "$LSP_BIN" 2>/dev/null || true)

if ! printf '%s' "$RESPONSE" | grep -q '"id":2'; then
    echo "FAIL: I did not answer the match-guard hover request"
    echo "$RESPONSE"
    exit 1
fi
if ! printf '%s' "$RESPONSE" | grep -q 'fn ready'; then
    echo "FAIL: I did not resolve the function used only by the match guard"
    echo "$RESPONSE"
    exit 1
fi

echo "PASS: I resolve identifiers inside match guards"
