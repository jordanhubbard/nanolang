#!/usr/bin/env bash
# ============================================================================
# Public make contract: `make launcher` compiles before it launches
# ============================================================================
#
# Observable interface: `make -C examples launcher`. When pkg-config has
# not yet found SDL2, `build` used to skip sdl_example_launcher and the
# launcher recipe still exec'd it (Error 127). The contract is that a
# dry-run still plans to compile sdl_example_launcher.nano, and that
# compile is planned before "Launching...".
#
# The compiler is a stub: this test is about Make's plan, not about
# compiling NanoLang.
#
# Usage:
#   bash tests/test_launcher_makefile.sh
# ============================================================================

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

FAILURES=0
pass() { echo -e "${GREEN}✅${NC} $1"; }
fail() { echo -e "${RED}❌${NC} $1"; FAILURES=$((FAILURES + 1)); }

echo "=========================================="
echo "Example launcher Makefile tests"
echo "=========================================="

STUB_DIR="$(mktemp -d)"
trap 'rm -rf "$STUB_DIR"' EXIT
STUB_COMPILER="$STUB_DIR/nanoc_c"
printf '#!/bin/sh\nexit 0\n' >"$STUB_COMPILER"
chmod +x "$STUB_COMPILER"

PLAN="$(make -C examples -n launcher SDL_AVAILABLE=no COMPILER="$STUB_COMPILER" 2>&1)" || true

compile_line="$(echo "$PLAN" | grep -n 'sdl_example_launcher\.nano' | head -1 | cut -d: -f1)"
launch_line="$(echo "$PLAN" | grep -n 'Launching\.\.\.' | head -1 | cut -d: -f1)"

if [ -n "$compile_line" ] && [ -n "$launch_line" ] && [ "$compile_line" -lt "$launch_line" ]; then
    pass "make launcher plans to compile sdl_example_launcher before launching, even without SDL2"
else
    fail "make launcher does not plan compile-then-launch when SDL_AVAILABLE=no"
    echo "compile_line=${compile_line:-none} launch_line=${launch_line:-none}"
    echo "$PLAN" | tail -25
fi

echo "=========================================="
if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}All example launcher Makefile tests passed${NC}"
    exit 0
fi
echo -e "${RED}${FAILURES} example launcher Makefile test(s) failed${NC}"
exit 1
