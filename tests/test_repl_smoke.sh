#!/bin/bash
# Smoke test for nanolang REPL

set -e
REPL="${1:-./bin/nanolang-repl}"

if [ ! -x "$REPL" ]; then
    echo "SKIP: $REPL not found" >&2
    exit 0
fi

PASS=0
FAIL=0

check() {
    local desc="$1"
    local input="$2"
    local expected="$3"
    local actual
    actual=$(printf '%s\n:quit\n' "$input" | timeout 5 "$REPL" 2>&1)
    if echo "$actual" | grep -qF "$expected"; then
        echo "PASS: $desc"
        PASS=$((PASS+1))
    else
        echo "FAIL: $desc"
        echo "  Expected to contain: $expected"
        echo "  Got: $(echo "$actual" | head -5 | tr '\n' '|')"
        FAIL=$((FAIL+1))
    fi
}

# Basic evaluation
check "integer literal"   "42;"              "42"
check "arithmetic"        "1 + 2 * 3;"       "7"
check "let binding"       "let x = 100; x;" "100"
check "boolean"           "true;"            "true"
check "negation"          "10 - 3;"          "7"

# Meta-commands
check ":env command"      "let y = 99; :env" "y"
check ":help command"     ":help"            "quit"

# String result
check "string literal"    '"hello";'         '"hello"'

# ── Hot-reload commands ──────────────────────────────────────────────────────

# :load — load a file and eval it
TMPLOAD=$(mktemp /tmp/test_load_XXXXXX.nano)
echo 'fn greet() -> int { return 42 }' > "$TMPLOAD"
check ":load file"  ":load $TMPLOAD" "loaded"
rm -f "$TMPLOAD"

# :save — save session to file, verify it contains a known fragment
TMPSAVE=$(mktemp /tmp/test_save_XXXXXX.nano)
check ":save file"  "let saved_var = 7; :save $TMPSAVE" "saved"
rm -f "$TMPSAVE"

# :reload — define a function, write an updated version to disk, reload it
TMPRELOAD=$(mktemp /tmp/test_reload_XXXXXX.nano)
echo 'fn hotfn() -> int { return 1 }' > "$TMPRELOAD"
check ":reload hot-patch"  ":load $TMPRELOAD
fn hotfn() -> int { return 2 }
:reload $TMPRELOAD" "reload"
rm -f "$TMPRELOAD"

# :help should mention the new commands
check ":help mentions :load"   ":help" ":load"
check ":help mentions :reload" ":help" ":reload"
check ":help mentions :save"   ":help" ":save"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ]
