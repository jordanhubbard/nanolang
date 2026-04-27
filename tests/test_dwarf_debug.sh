#!/usr/bin/env bash
# test_dwarf_debug.sh — tests for nanoc --debug DWARF debug info emission
# Tests:
#   1. nanoc --debug compiles without error
#   2. DWARF v4 metadata present in binary (DW_AT_producer with -gdwarf-4)
#   3. .nano source file name appears in DWARF decoded line table
#   4. Line numbers from .nano source appear in line table
#   5. Generated C contains #line directives pointing to .nano source
#   6. Interpreter: shadow tests pass on the test file
#   7. --debug output is larger than non-debug (DWARF sections added)

# pipefail disabled: grep -m1 causes SIGPIPE in pipes which triggers pipefail
set -uo errexit
NANOC="${1:-./bin/nanoc}"
NANO="${2:-./bin/nano}"
PASS=0; FAIL=0
SRC="tests/unit/test_dwarf_debug.nano"

ok()  { echo "  ✅ $1"; PASS=$((PASS+1)); }
fail(){ echo "  ❌ $1: $2"; FAIL=$((FAIL+1)); }

echo "DWARF debug info tests (nanoc --debug):"

# ── 1. Interpreter: shadow tests pass ──────────────────────────────────
if "$NANO" "$SRC" > /tmp/dwarf_interp.txt 2>&1; then
    ok "interpreter: all shadow tests pass"
else
    fail "interpreter: shadow tests failed" "$(cat /tmp/dwarf_interp.txt | head -3)"
fi

# ── 2. nanoc --debug compiles cleanly ──────────────────────────────────
if "$NANOC" "$SRC" --debug -o /tmp/test_debug_out > /tmp/dwarf_compile.txt 2>&1; then
    ok "nanoc --debug: compiles cleanly"
else
    fail "nanoc --debug: compilation failed" "$(cat /tmp/dwarf_compile.txt | grep -v Warning | head -3)"
fi

# ── 3. Binary contains DWARF info (objdump --dwarf=info) ───────────────
if [ -f /tmp/test_debug_out ]; then
    DBG_STRINGS=$(strings /tmp/test_debug_out 2>/dev/null || true)
    if echo "$DBG_STRINGS" | grep -q "gdwarf-4"; then
        ok "DWARF v4 producer string present (contains -gdwarf-4)"
    else
        fail "DWARF v4 producer string missing" "strings output lacks -gdwarf-4"
    fi
else
    fail "debug binary not created" ""
fi

# ── 4. .nano file name in DWARF line table ──────────────────────────────
if [ -f /tmp/test_debug_out ]; then
    BASENAME=$(basename "$SRC")
    if echo "$DBG_STRINGS" | grep -q "$BASENAME"; then
        ok ".nano source name ($BASENAME) appears in DWARF line table"
    else
        fail ".nano source name missing from DWARF line table" \
             "$(objdump --dwarf=decodedline /tmp/test_debug_out 2>/dev/null | head -5)"
    fi
fi

# ── 5. Line numbers from .nano source in line table ─────────────────────
if [ -f /tmp/test_debug_out ]; then
    LINE_COUNT=$(objdump --dwarf=decodedline /tmp/test_debug_out 2>/dev/null | \
                  grep "$(basename $SRC)" | wc -l)
    if [ "$LINE_COUNT" -ge 5 ]; then
        ok "DWARF line table has $LINE_COUNT .nano source line entries"
    else
        fail "Too few .nano line entries in DWARF" "got $LINE_COUNT (expected ≥5)"
    fi
fi

# ── 6. Generated C has #line directives (--keep-c) ─────────────────────
if "$NANOC" "$SRC" --debug --keep-c -o /tmp/test_debug_out2 > /dev/null 2>&1; then
    CFILE="/tmp/test_debug_out2.c"
    if [ -f "$CFILE" ] && grep -q "^#line.*test_dwarf_debug.nano" "$CFILE"; then
        LINE_DIRECTIVES=$(grep -c "^#line" "$CFILE" || true)
        ok "#line directives in generated C ($LINE_DIRECTIVES total)"
    else
        fail "#line directives missing in generated C" "$(ls /tmp/test_debug_out2.c 2>/dev/null || echo 'no .c file')"
    fi
fi

# ── 7. Without --debug: no DWARF line entries for .nano source ──────────
if "$NANOC" "$SRC" -o /tmp/test_nodebug_out > /dev/null 2>&1; then
    BASENAME=$(basename "$SRC")
    if ! objdump --dwarf=decodedline /tmp/test_nodebug_out 2>/dev/null | grep -q "$BASENAME"; then
        ok "without --debug: no .nano entries in DWARF (as expected)"
    else
        fail "without --debug: unexpected .nano entries in DWARF" ""
    fi
fi

# ── 8. Debug binary larger than non-debug ──────────────────────────────
if [ -f /tmp/test_debug_out ] && [ -f /tmp/test_nodebug_out ]; then
    SZ_DBG=$(wc -c < /tmp/test_debug_out)
    SZ_NOD=$(wc -c < /tmp/test_nodebug_out)
    if [ "$SZ_DBG" -gt "$SZ_NOD" ]; then
        ok "--debug binary larger ($SZ_DBG vs $SZ_NOD bytes, DWARF sections added)"
    else
        fail "--debug binary not larger than non-debug" "$SZ_DBG vs $SZ_NOD bytes"
    fi
fi

rm -f /tmp/dwarf_interp.txt /tmp/dwarf_compile.txt \
       /tmp/test_debug_out /tmp/test_debug_out.c \
       /tmp/test_debug_out2 /tmp/test_debug_out2.c \
       /tmp/test_nodebug_out

echo ""
echo "Results: $PASS passed, $FAIL failed"
[ $FAIL -eq 0 ]
