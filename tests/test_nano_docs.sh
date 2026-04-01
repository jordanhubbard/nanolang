#!/usr/bin/env bash
# test_nano_docs.sh — tests for the nano-docs search tool
NANO_DOCS="${1:-./bin/nano-docs}"
PASS=0; FAIL=0

check() {
    local desc="$1"; local got="$2"; local pattern="$3"
    if echo "$got" | grep -q "$pattern"; then
        echo "  ✅ $desc"; PASS=$((PASS+1))
    else
        echo "  ❌ $desc (expected pattern: $pattern)"; FAIL=$((FAIL+1))
    fi
}

echo "nano-docs tests:"

# ── Basic search ──────────────────────────────────────────────────────
OUT=$("$NANO_DOCS" "str_concat" --no-color 2>&1)
check "finds str_concat in modules" "$OUT" "str_concat"
check "returns result count"        "$OUT" "result"

# ── --docs mode ────────────────────────────────────────────────────────
OUT2=$("$NANO_DOCS" "string" --docs --no-color 2>&1)
check "finds 'string' in docs"     "$OUT2" "string"
check "docs mode returns results"  "$OUT2" "result"

# ── --fn filter ────────────────────────────────────────────────────────
OUT3=$("$NANO_DOCS" --fn "println" --no-color 2>&1)
check "finds println fn declaration" "$OUT3" "println"

# ── --list-modules ──────────────────────────────────────────────────────
OUT4=$("$NANO_DOCS" --list-modules --no-color 2>&1)
check "lists std module"       "$OUT4" "std"
check "lists readline module"  "$OUT4" "readline"

# ── case-insensitive ────────────────────────────────────────────────────
OUT5=$("$NANO_DOCS" -i "async" --no-color 2>&1)
check "case-insensitive finds async" "$OUT5" "result"

# ── no results case ─────────────────────────────────────────────────────
"$NANO_DOCS" "xyzzy_not_found_ever_12345" --no-color 2>&1 | grep -q "No results" && {
    echo "  ✅ no-results message shown"; PASS=$((PASS+1))
} || {
    echo "  ❌ should show 'No results' for unknown query"; FAIL=$((FAIL+1))
}

# ── --module filter ──────────────────────────────────────────────────────
OUT6=$("$NANO_DOCS" --module std "maybe" --no-color 2>&1)
check "module filter works for std/maybe" "$OUT6" "maybe"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ]
