#!/usr/bin/env bash
# ============================================================================
# Module system dependencies are re-checked when the object cache is warm
# ============================================================================
#
# install_system_packages() and the pkg-config verification used to live inside
# `if (needs_rebuild)`, so a module whose cached object was still current never
# had its dependencies checked. On a host missing the headers the build died in
# the C compiler ("fatal error: SDL2/SDL_ttf.h: No such file or directory")
# instead of installing the package or naming the missing dependency.
#
# Whether a dev package is installed is a fact about the machine, not about my
# object cache, so the check must happen on every build.
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
  "description": "Fixture module guarded by a stub pkg-config package",
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
/* nanolang's int is int64_t; long long would conflict on LP64 Linux. */
int64_t fakedep_answer(void);
#endif
EOF

cat >"$MOD_DIR/fakedep.c" <<'EOF'
#include "fakedep.h"
int64_t fakedep_answer(void) { return 42; }
EOF

cat >"$MOD_DIR/fakedep.nano" <<'EOF'
# Fixture module: one extern function backed by fakedep.c
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

# A real .pc file stands in for the installed dev package. Deleting it later
# is how this test simulates "the package is not on this host".
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

# --- 1. warm the object cache while the dependency is present ---------------
BUILD1="$(perl -e 'alarm 180; exec @ARGV' "$COMPILER" "$WORK_REL/prog.nano" -o "$WORK/prog" 2>&1)"
BUILD1_STATUS=$?

if [ "$BUILD1_STATUS" -eq 0 ]; then
    pass "fixture module builds while its pkg-config package is present"
else
    fail "fixture module did not build with its dependency present"
    echo "$BUILD1" | tail -20
    echo "=========================================="
    echo -e "${RED}${FAILURES} module dependency re-check test(s) failed${NC}"
    exit 1
fi

CACHED_OBJ="$(find "$NANO_BUILD_CACHE" -name 'fakedep.o' -print -quit 2>/dev/null)"
if [ -n "$CACHED_OBJ" ]; then
    pass "module object is cached in NANO_BUILD_CACHE"
else
    fail "no cached fakedep.o - test cannot exercise the warm-cache path"
fi

# --- 2. the dependency disappears but the cached object stays ---------------
rm -f "$PC_FILE"

BUILD2="$(perl -e 'alarm 180; exec @ARGV' "$COMPILER" "$WORK_REL/prog.nano" -o "$WORK/prog2" 2>&1)"
BUILD2_STATUS=$?

if [ "$BUILD2_STATUS" -ne 0 ]; then
    pass "build fails when a cached module's dependency is missing"
else
    fail "build succeeded with a missing dependency - the warm cache skipped the check"
fi

if echo "$BUILD2" | grep -q "not found for module 'fakedep'"; then
    pass "missing dependency is reported by name, not left to the C compiler"
else
    fail "no dependency diagnostic for the missing package"
    echo "$BUILD2" | tail -20
fi

echo "=========================================="
if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}All module dependency re-check tests passed${NC}"
    exit 0
fi
echo -e "${RED}${FAILURES} module dependency re-check test(s) failed${NC}"
exit 1
