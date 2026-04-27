#!/usr/bin/env bash
# test_docgen_md.sh — verify --doc-md GFM Markdown docgen
#
# Usage: bash tests/test_docgen_md.sh [path/to/nanoc_c]
set -euo pipefail

NANOC="${1:-./bin/nanoc_c}"
NANO_FILE="tests/docgen/example.nano"
OUT_FILE="$(mktemp /tmp/test_docgen_md_XXXXXX.md)"
PASS=0
FAIL=0

cleanup() { rm -f "$OUT_FILE"; }
trap cleanup EXIT

check() {
    local desc="$1" pattern="$2"
    if grep -qe "$pattern" "$OUT_FILE"; then
        echo "  PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $desc  (expected pattern: $pattern)"
        FAIL=$((FAIL + 1))
    fi
}

no_match() {
    local desc="$1" pattern="$2"
    if grep -qe "$pattern" "$OUT_FILE"; then
        echo "  FAIL: $desc  (unexpected pattern found: $pattern)"
        FAIL=$((FAIL + 1))
    else
        echo "  PASS: $desc"
        PASS=$((PASS + 1))
    fi
}

echo "=== test_docgen_md.sh ==="
echo "Compiler: $NANOC"
echo "Input:    $NANO_FILE"
echo "Output:   $OUT_FILE"
echo ""

# Run nanoc --doc-md
"$NANOC" "$NANO_FILE" --doc-md -o "$OUT_FILE"

echo "Checking output..."

# 1. Top-level module heading
check "module heading exists" "^# nano stdlib"

# 2. Function headings use ##  with backtick signature
check "fn add heading" "^## \`pub fn add"
check "fn multiply heading" "^## \`pub fn multiply"
check "fn is_even heading" "^## \`pub fn is_even"

# 3. Parameter types appear in signature
check "param types in add sig" "int"

# 4. Return type in signature
check "return type -> int" "[>] int"
check "return type -> bool" "[>] bool"

# 5. Doc body text
check "add doc text" "Add two integers together"
check "multiply doc text" "Multiply two integers"

# 6. @param becomes parameter table
check "parameters table header" "\*\*Parameters:\*\*"
check "param table row for a" "| \`a\`"
check "param table row for b" "| \`b\`"

# 7. @returns becomes Returns line
check "@returns becomes Returns:" "\*\*Returns:\*\*"

# 8. @example block becomes fenced nano code block
check "example fenced code block" '```nano'
check "example content" "let result"

# 9. Struct heading
check "struct Point heading" "^## \`pub struct Point\`"
check "struct fields table" "\*\*Fields:\*\*"
check "struct field x" "| \`x\`"
check "struct field y" "| \`y\`"

# 10. Enum heading and variants
check "enum Color heading" "^## \`pub enum Color\`"
check "enum variants header" "\*\*Variants:\*\*"
check "enum variant Red" "[\-] .Red."

# 11. Union heading and variants
check "union Maybe heading" "^## \`pub union Maybe\`"
check "union variant Some" "[\-] .Some"
check "union variant None" "[\-] .None"

# 12. Horizontal rules between declarations
check "horizontal rule exists" "^---$"

# 13. No raw HTML in output
no_match "no <html> tag" "<html"
no_match "no <div> tag" "<div"
no_match "no <span> tag" "<span"
no_match "no &lt; entity" "&lt;"
no_match "no &amp; entity" "&amp;"

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
