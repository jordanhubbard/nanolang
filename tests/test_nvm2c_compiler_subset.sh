#!/usr/bin/env bash
# nvm2c covers the compiler subset: C-seed nanoc_v06.nvm -> C11 -> cc,
# no nano_vm, --help runs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/nanolang-nvm2c-nanoc.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

TIMEOUT_EMIT=(perl -e 'alarm 180; exec @ARGV')
TIMEOUT_XLATE=(perl -e 'alarm 240; exec @ARGV')
TIMEOUT_CC=(perl -e 'alarm 180; exec @ARGV')

NANO_VIRT="$PROJECT_ROOT/bin/nano_virt"
NVM2C="$PROJECT_ROOT/bin/nvm2c"
NANOC_NANO="$PROJECT_ROOT/src_nano/nanoc_v06.nano"
NVM="${NANOLANG_NANOC_NVM:-}"

need() {
    if [ ! -x "$1" ]; then
        echo "ERROR: missing $1" >&2
        exit 1
    fi
}

need "$NVM2C"
if [ -z "$NVM" ]; then
    need "$NANO_VIRT"
    if [ ! -f "$NANOC_NANO" ]; then
        echo "ERROR: missing $NANOC_NANO" >&2
        exit 1
    fi
    echo "C-seed --emit-nvm of nanoc_v06..."
    export NANO_MODULE_PATH="$PROJECT_ROOT/modules"
    "${TIMEOUT_EMIT[@]}" "$NANO_VIRT" "$NANOC_NANO" --emit-nvm --strip-debug -o "$WORK/nanoc.nvm"
    NVM="$WORK/nanoc.nvm"
fi
if [ ! -f "$NVM" ]; then
    echo "ERROR: missing nanoc .nvm at $NVM" >&2
    exit 1
fi

echo "nvm2c of nanoc_v06..."
"${TIMEOUT_XLATE[@]}" "$NVM2C" "$NVM" -o "$WORK/nanoc.c"
test -s "$WORK/nanoc.c"
if grep -E 'nano_vm' "$WORK/nanoc.c" >/dev/null; then
    echo "ERROR: nvm2c C names nano_vm" >&2
    exit 1
fi

echo "cc -std=c11 of nvm2c nanoc..."
"${TIMEOUT_CC[@]}" cc -std=c11 -Wall -Wextra -Werror -O0 -o "$WORK/nanoc.bin" "$WORK/nanoc.c"
need "$WORK/nanoc.bin"

help_out="$("$WORK/nanoc.bin" --help)"
case "$help_out" in
    *"Self-Hosted Compiler"*) ;;
    *)
        echo "ERROR: AOT nanoc --help did not print usage" >&2
        exit 1
        ;;
esac

echo "nvm2c compiler subset (nanoc_v06): ok"
