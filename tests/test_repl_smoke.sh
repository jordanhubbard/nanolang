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

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ]
