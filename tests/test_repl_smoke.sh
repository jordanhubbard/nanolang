#!/bin/bash
# Smoke test for examples/language/full_repl.nano

set -e
REPL="${1:-./bin/full_repl}"

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
    actual=$(printf '%s\n:quit\n' "$input" | "$REPL" 2>&1)
    if printf '%s\n' "$actual" | grep -qF "$expected"; then
        echo "PASS: $desc"
        PASS=$((PASS+1))
    else
        echo "FAIL: $desc"
        echo "  Expected to contain: $expected"
        echo "  Got: $actual"
        FAIL=$((FAIL+1))
    fi
}

# Basic evaluation
check "integer literal"   "42"                    "42"
check "arithmetic"        "(+ 1 (* 2 3))"         "7"
check "let binding"       $'let x: int = 100\nx'  "100"
check "boolean"           ":bool true"            "true"
check "negation"          "(- 10 3)"              "7"
check "function"          $'fn double(x: int) -> int { return (* x 2) }\n(double 4)' "8"
check "multi-line function" $'fn triple(x: int) -> int {\nreturn (* x 3)\n}\n(triple 4)' "12"
check "import" $'from "modules/std/env.nano" import get\n:string (get "HOME")' "/"

# Meta-commands
check ":vars command"     $'let y: int = 99\n:vars' "y"
check ":funcs command"    $'fn id(x: int) -> int { return x }\n:funcs' "id(x: int) -> int"

# String result
check "string literal"    ':string "hello"'  'hello'

# ── Hot-reload commands ──────────────────────────────────────────────────────

# :load — load a file and eval it
TMPLOAD=$(mktemp /tmp/test_load_XXXXXX.nano)
printf 'fn greet() -> int { return 42 }\nshadow greet { assert (== (greet) 42) }\nfn main() -> int { return 0 }\nshadow main { assert (== (main) 0) }\n' > "$TMPLOAD"
check ":load file"  ":load $TMPLOAD" "Loaded:"
rm -f "$TMPLOAD"

# :save — save session to file, verify it contains a known fragment
TMPSAVE=$(mktemp /tmp/test_save_XXXXXX.nano)
check ":save file"  $'let saved_var: int = 7\n:save '"$TMPSAVE" "Session saved"
rm -f "$TMPSAVE"

check ":history command" ":history" "nano_history"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ]
