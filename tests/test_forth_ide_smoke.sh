#!/usr/bin/env bash
# Build smoke for sdl_forth_ide. Interpreter liveness is make test-forth-pty.
# Graphical init uses xvfb when present; missing SDL skips the compile.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [ ! -x bin/forth ]; then
    echo "FAIL bin/forth is not executable" >&2
    exit 1
fi
echo "  ✓ bin/forth exists (PTY child for sdl_forth_ide)"

sdl_yes=no
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists sdl2 2>/dev/null; then
    sdl_yes=yes
elif [ -x /opt/homebrew/bin/pkg-config ] && /opt/homebrew/bin/pkg-config --exists sdl2 2>/dev/null; then
    sdl_yes=yes
fi

if [ "$sdl_yes" != yes ]; then
    echo "  ⊘ sdl_forth_ide compile skipped (sdl2 pkg-config missing)"
    exit 0
fi

compiler=""
if [ -x bin/nanoc_c ]; then
    compiler=../bin/nanoc_c
elif [ -x bin/nanoc ]; then
    compiler=../bin/nanoc
else
    echo "  ⊘ sdl_forth_ide compile skipped (nanoc not built)"
    exit 0
fi

export NANO_MODULE_PATH="$repo_root/modules"
perl -e 'alarm 180; exec @ARGV' make -C examples "../bin/sdl_forth_ide" \
    "COMPILER=$compiler" EXAMPLES_BACKEND=c \
    NANO_MODULE_PATH="$repo_root/modules"

if [ ! -x bin/sdl_forth_ide ]; then
    echo "FAIL bin/sdl_forth_ide was not built" >&2
    exit 1
fi
echo "  ✓ bin/sdl_forth_ide compiled"

if command -v xvfb-run >/dev/null 2>&1 && command -v timeout >/dev/null 2>&1; then
    set +e
    xvfb-run -a timeout 3 bin/sdl_forth_ide >/tmp/sdl_forth_ide_smoke.log 2>&1
    status=$?
    set -e
    case "$status" in
        124|143)
            echo "  ✓ sdl_forth_ide survived graphical initialization"
            ;;
        *)
            echo "FAIL sdl_forth_ide exited during initialization (status $status)" >&2
            tail -20 /tmp/sdl_forth_ide_smoke.log >&2 || true
            exit 1
            ;;
    esac
else
    echo "  ⊘ graphical initialization requires xvfb-run and timeout"
fi
