#!/usr/bin/env bash
# ============================================================================
# Example launcher Makefile — do not exec a skipped SDL binary
# ============================================================================
#
# `make launcher` used to run ./bin/sdl_example_launcher after `build` even
# when SDL_AVAILABLE was no, so Linux hosts without libsdl2-dev printed
# "Launching..." and then Error 127 / not found. The launcher target must
# name that binary as a make prerequisite so compiling it can pull in the
# SDL modules (and auto-install) instead of exec'ing a file that was skipped.
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

if echo "$PLAN" | grep -q 'sdl_example_launcher\.nano'; then
    pass "launcher still compiles sdl_example_launcher when SDL_AVAILABLE=no"
else
    fail "launcher dry-run skipped sdl_example_launcher.nano when SDL_AVAILABLE=no"
    echo "$PLAN" | tail -20
fi

compile_line="$(echo "$PLAN" | grep -n 'sdl_example_launcher\.nano' | head -1 | cut -d: -f1)"
launch_line="$(echo "$PLAN" | grep -n 'Launching\.\.\.' | head -1 | cut -d: -f1)"
if [ -n "$compile_line" ] && [ -n "$launch_line" ] && [ "$compile_line" -lt "$launch_line" ]; then
    pass "compile of sdl_example_launcher is planned before Launching"
else
    fail "Launching is not sequenced after compiling sdl_example_launcher"
    echo "compile_line=${compile_line:-none} launch_line=${launch_line:-none}"
fi

if grep -q 'launcher: build $(LAUNCHER_BIN)' examples/Makefile \
    || grep -q 'launcher: build \$(LAUNCHER_BIN)' examples/Makefile \
    || grep -E '^launcher:.*sdl_example_launcher' examples/Makefile >/dev/null; then
    pass "examples/Makefile launcher prerequisite names the SDL launcher binary"
else
    fail "examples/Makefile launcher target no longer depends on the launcher binary"
    grep -n '^launcher:' examples/Makefile
fi

echo "=========================================="
if [ "$FAILURES" -eq 0 ]; then
    echo -e "${GREEN}All example launcher Makefile tests passed${NC}"
    exit 0
fi
echo -e "${RED}${FAILURES} example launcher Makefile test(s) failed${NC}"
exit 1
