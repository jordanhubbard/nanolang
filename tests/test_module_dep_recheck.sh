#!/usr/bin/env bash
# ============================================================================
# Compiler contract: a cached module still requires its system packages
# ============================================================================
#
# Public interface: `nanoc_c` compiling a program that imports a C module
# with pkg_config. Whether the host still has that package is independent
# of whether I already have an object in NANO_BUILD_CACHE.
#
# Scenario: compile while the package is present → run the binary → the
# package disappears → compile again. The second compile must fail with my
# named missing-package diagnostic, not a C compiler "No such file" on a
# header. Restoring the package must compile again.
#
# Usage:
#   bash tests/test_module_dep_recheck.sh
# ============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

FAILURES=0
pass() { echo -e "${GREEN}✅${NC} $1"; }
fail() { echo -e "${RED}❌${NC} $1"; FAILURES=$((FAILURES + 1)); }
skip() { echo -e "${YELLOW}⊘${NC} $1"; }

echo "=========================================="
echo "Module dependency re-check tests"
echo "=========================================="

COMPILER="$PROJECT_ROOT/bin/nanoc_c"
if [ ! -x "$COMPILER" ]; then
    skip "bin/nanoc_c not built - run 'make' first"
    exit 0
fi

if ! command -v pkg-config >/dev/null 2>&1; then
    skip "pkg-config not installed"
    exit 0
fi

WORK="$(mktemp -d "$PROJECT_ROOT/.tmp_dep_recheck.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
WORK_REL="$(basename "$WORK")"

MOD_DIR="$WORK/modules/fakedep"
mkdir -p "$MOD_DIR" "$WORK/pc" "$WORK/cache"

cat >"$MOD_DIR/module.json" <<EOF
{
  "name": "fakedep",
  "version": "1.0.0",
  "description": "Fixture module guarded by a pkg-config package",
  "headers": ["fakedep.h"],
  "cflags": ["-I$WORK_REL/modules/fakedep"],
  "c_sources": ["fakedep.c"],
  "pkg_config": ["fakedep"]
}
EOF

cat >"$MOD_DIR/fakedep.h" <<'EOF'
#ifndef FAKEDEP_H
#define FAKEDEP_H
#include <stdint.h>
int64_t fakedep_answer(void);
#endif
EOF

cat >"$MOD_DIR/fakedep.c" <<'EOF'
#include "fakedep.h"
int64_t fakedep_answer(void) { return 42; }
EOF

cat >"$MOD_DIR/fakedep.nano" <<'EOF'
extern fn fakedep_answer() -> int
EOF

cat >"$WORK/prog.nano" <<EOF
unsafe module "$WORK_REL/modules/fakedep/fakedep.nano"

fn main() -> int {
    let answer: int = (fakedep_answer)
    (print (int_to_string answer))
    return 0
}

shadow main {
    assert (== (fakedep_answer) 42)
}
EOF

PC_FILE="$WORK/pc/fakedep.pc"
cat >"$PC_FILE" <<'EOF'
Name: fakedep
Description: fixture package for the module dependency re-check test
Version: 1.0.0
Cflags:
Libs:
EOF

export PKG_CONFIG_PATH="$WORK/pc"
export NANO_BUILD_CACHE="$WORK/cache"

compile() {
    local out="$1"
    perl -e 'alarm 180; exec @ARGV' "$COMPILER" "$WORK_REL/prog.nano" -o "$out" 2>&1
}

BUILD1="$(compile "$WORK/prog")"
BUILD1_STATUS=$?

if [ "$BUILD1_STATUS" -ne 0 ]; then
    fail "compile failed while the pkg-config package was present"
    echo "$BUILD1" | tail -20
    echo "=========================================="
    echo -e "${RED}${FAILURES} module dependency re-check test(s) failed${NC}"
    exit 1
fi

if [ ! -x "$WORK/prog" ]; then
    fail "compile reported success but wrote no binary"
    echo "=========================================="
    echo -e "${RED}${FAILURES} module dependency re-check test(s) failed${NC}"
    exit 1
fi

RUN_OUT="$("$WORK/prog" 2>&1)"
RUN_STATUS=$?
if [ "$RUN_STATUS" -eq 0 ] && echo "$RUN_OUT" | grep -q '42'; then
    pass "compile and run succeed while the system package is present"
else
    fail "compiled binary did not print 42 (exit $RUN_STATUS)"
    echo "$RUN_OUT"
fi

# The warm-cache failure mode only exists if the first compile actually
# cached an object. That is a scenario precondition, not a product claim.
if ! find "$NANO_BUILD_CACHE" -name 'fakedep.o' -print -quit | grep -q .; then
    fail "setup: first compile wrote no cached object; cannot observe a warm-cache rebuild"
    echo "=========================================="
    echo -e "${RED}${FAILURES} module dependency re-check test(s) failed${NC}"
    exit 1
fi

rm -f "$PC_FILE"

BUILD2="$(compile "$WORK/prog2")"
BUILD2_STATUS=$?

if [ "$BUILD2_STATUS" -eq 0 ]; then
    fail "compile succeeded after the system package disappeared"
else
    pass "compile fails when the cached module's system package is gone"
fi

if echo "$BUILD2" | grep -q "not found for module 'fakedep'"; then
    pass "failure names the missing package"
else
    fail "failure did not name the missing package"
    echo "$BUILD2" | tail -20
fi

if echo "$BUILD2" | grep -qi 'fatal error:'; then
    fail "failure was a C compiler fatal error instead of a package diagnostic"
    echo "$BUILD2" | tail -20
else
    pass "failure is my diagnostic, not a missing-header C error"
fi

cat >"$PC_FILE" <<'EOF'
Name: fakedep
Description: fixture package for the module dependency re-check test
Version: 1.0.0
Cflags:
Libs:
EOF

BUILD3="$(compile "$WORK/prog3")"
BUILD3_STATUS=$?
if [ "$BUILD3_STATUS" -eq 0 ] && [ -x "$WORK/prog3" ]; then
    pass "compile recovers after the system package is restored"
else
    fail "compile still fails after restoring the system package"
    echo "$BUILD3" | tail -20
fi

echo "=========================================="
if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}All module dependency re-check tests passed${NC}"
    exit 0
fi
echo -e "${RED}${FAILURES} module dependency re-check test(s) failed${NC}"
exit 1
