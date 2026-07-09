#!/usr/bin/env bash
# test_script_mode.sh — tests for nano --script and nano -e
NANO="${1:-./bin/nano}"
PASS=0; FAIL=0

check() {
    local desc="$1" expected="$2"
    local got="$(eval "${3}" 2>&1)"
    if echo "$got" | grep -qF "$expected"; then
        echo "  ✅ $desc"; PASS=$((PASS+1))
    else
        echo "  ❌ $desc (expected '$expected', got: $got)"; FAIL=$((FAIL+1))
    fi
}

echo "nano script mode tests:"

# ── -e expression mode ──────────────────────────────────────────────────
check "-e: simple arithmetic"        "3" \
    '"$NANO" -e "(+ 1 2)"'

check "-e: string expression"        "hello" \
    '"$NANO" -e '"'"'(str_concat "hel" "lo")'"'"''

check "-e: boolean"                  "true" \
    '"$NANO" -e "(== 1 1)"'

check "-e: fn statement"             "" \
    '"$NANO" -e "fn noop() -> int { return 0 }"'

# ── --script file mode ──────────────────────────────────────────────────
TMP=$(mktemp /tmp/nano_script_XXXXXX.nano)
cat > "$TMP" << 'NANO'
fn double(x: int) -> int { return (* x 2) }
shadow double { assert (== (double 5) 10) }
(println (str_concat "double(21)=" (to_string (double 21))))
NANO
check "--script file: fn + call"     "double(21)=42" \
    '"$NANO" --script "$TMP"'

# ── --script stdin mode ─────────────────────────────────────────────────
check "--script -: stdin pipe"       "sum=55" \
"printf 'fn sum_to(n: int) -> int {\n    let mut t: int = 0\n    let mut i: int = 1\n    while (<= i n) {\n        set t (+ t i)\n        set i (+ i 1)\n    }\n    return t\n}\nshadow sum_to { assert (== (sum_to 10) 55) }\n(println (str_concat \"sum=\" (to_string (sum_to 10))))\n' | \"$NANO\" --script -"

# ── --script with existing main ─────────────────────────────────────────
TMP2=$(mktemp /tmp/nano_script_main_XXXXXX.nano)
cat > "$TMP2" << 'NANO'
fn main() -> int {
    (println "script-with-main OK")
    return 0
}
shadow main { assert (== (main) 0) }
NANO
check "--script: respects existing main" "script-with-main OK" \
    '"$NANO" --script "$TMP2"'

rm -f "$TMP" "$TMP2"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ]
