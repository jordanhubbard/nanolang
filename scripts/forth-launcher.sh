#!/usr/bin/env bash
# =============================================================================
# Nano Forth launcher (bin/forth)
# =============================================================================
# `bin/forth` is a thin PTY-friendly front end for the ONE Forth implementation:
# the NanoISA-backed `nl_forth_interpreter` compiled to verified NanoISA
# bytecode (`bin/nl_forth_interpreter_vm`) and executed by `bin/nano_vm`.
#
# It is intentionally NOT a second, separately compiled native Forth binary.
# The SDL IDE (examples/graphics/sdl_forth_ide.nano) launches this script over a
# PTY, so anything that reaches the user through the IDE runs on the same
# NanoISA-backed executable the CLI and the ANS test suite use.
#
# Interactivity is selected by the interpreter itself: it starts the REPL when
# stdin is a TTY (as it is under the IDE's PTY) or when FORTH_INTERACTIVE=1.
# `--interactive` is accepted for backwards compatibility and simply forces the
# REPL on.
set -euo pipefail

# Resolve the repository root from this script's location so the launcher works
# regardless of the caller's working directory.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

vm="$repo_root/bin/nano_vm"
module="$repo_root/bin/nl_forth_interpreter_vm"
forth_see_lib="$repo_root/modules/forth_see/.build"

if [ ! -x "$vm" ]; then
    echo "bin/forth: NanoVM ($vm) is not built. Run 'make forth'." >&2
    exit 1
fi
if [ ! -f "$module" ]; then
    echo "bin/forth: NanoISA Forth module ($module) is not built. Run 'make forth'." >&2
    exit 1
fi

# Extern words (readline, SEE disassembly) are resolved by the co-process, which
# dlopens libforth_see from its build directory.
if [ -d "$forth_see_lib" ]; then
    export LD_LIBRARY_PATH="$forth_see_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    export DYLD_LIBRARY_PATH="$forth_see_lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
fi

# Honour an explicit --interactive request even when stdin is redirected.
for arg in "$@"; do
    if [ "$arg" = "--interactive" ]; then
        export FORTH_INTERACTIVE=1
    fi
done

exec "$vm" "$module"
