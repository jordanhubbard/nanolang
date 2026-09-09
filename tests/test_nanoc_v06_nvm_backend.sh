#!/usr/bin/env bash
# Self-hosted nanoc_v06 product is .nvm. Native binaries are nvm2c then cc.
# Stage 0 (nanoc_c) still pretty-prints C to *build* the driver.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPILER_C="${1:-$PROJECT_ROOT/bin/nanoc_c}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/nanolang-nanoc-v06-nvm.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

TIMEOUT=(perl -e 'alarm 600; exec @ARGV')
HELLO="$PROJECT_ROOT/examples/language/nl_hello.nano"
NANOISA_EMIT="$PROJECT_ROOT/bin/nanoisa_emit"
NANOISA="$PROJECT_ROOT/bin/nanoisa"
NVM2C="$PROJECT_ROOT/bin/nvm2c"

need() {
    if [ ! -x "$1" ]; then
        echo "ERROR: missing $1" >&2
        exit 1
    fi
}

need "$COMPILER_C"
need "$NANOISA_EMIT"
need "$NANOISA"
need "$NVM2C"
if [ ! -f "$HELLO" ]; then
    echo "ERROR: missing $HELLO" >&2
    exit 1
fi

echo "Checking nvm2c AOT of nl_hello (tool pipeline, no driver)..."
"${TIMEOUT[@]}" "$NANOISA_EMIT" "$HELLO" -o "$WORK/hello.nasm"
test -s "$WORK/hello.nasm"
if grep -F "I refused that program:" "$WORK/hello.nasm" >/dev/null; then
    echo "ERROR: nanoisa_emit refused nl_hello" >&2
    exit 1
fi
"${TIMEOUT[@]}" "$NANOISA" asm "$WORK/hello.nasm" -o "$WORK/hello.nvm"
test -s "$WORK/hello.nvm"
"${TIMEOUT[@]}" "$NVM2C" "$WORK/hello.nvm" -o "$WORK/hello.c"
test -s "$WORK/hello.c"
if grep -E 'nano_vm' "$WORK/hello.c" >/dev/null; then
    echo "ERROR: nvm2c C mentions nano_vm" >&2
    exit 1
fi
"${TIMEOUT[@]}" cc -std=c11 -Wall -Wextra -Werror -o "$WORK/hello.bin" "$WORK/hello.c"
out="$("$WORK/hello.bin")"
if [ "$out" != "Hello from NanoLang!" ]; then
    echo "ERROR: nvm2c hello output was [$out]" >&2
    exit 1
fi

echo "Compiling nanoc_v06.nano with C seed (Stage 0 still pretty-prints C)..."
export NANO_MODULE_PATH="$PROJECT_ROOT/modules"
"${TIMEOUT[@]}" "$COMPILER_C" "$PROJECT_ROOT/src_nano/nanoc_v06.nano" -o "$WORK/nanoc_v06"
need "$WORK/nanoc_v06"

echo "Checking nanoc_v06 --emit-nvm of nl_hello..."
"${TIMEOUT[@]}" "$WORK/nanoc_v06" "$HELLO" --emit-nvm -o "$WORK/hello_drv.nvm"
test -s "$WORK/hello_drv.nvm"

echo "Checking nanoc_v06 -o binary of nl_hello (nvm2c then cc)..."
"${TIMEOUT[@]}" "$WORK/nanoc_v06" "$HELLO" -o "$WORK/hello_drv.bin"
need "$WORK/hello_drv.bin"
drv_out="$("$WORK/hello_drv.bin")"
if [ "$drv_out" != "Hello from NanoLang!" ]; then
    echo "ERROR: driver hello output was [$drv_out]" >&2
    exit 1
fi

if grep -n 'transpile_parser' "$PROJECT_ROOT/src_nano/nanoc_v06.nano" >/dev/null; then
    echo "ERROR: nanoc_v06.nano still calls transpile_parser" >&2
    exit 1
fi

echo "nanoc_v06 NanoISA backend: ok"
