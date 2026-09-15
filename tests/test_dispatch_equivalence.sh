#!/usr/bin/env bash
#
# The two dispatch strategies must be the same interpreter.
#
# Computed goto and the portable switch share one copy of the handlers, so they
# cannot drift in what an instruction *does*. What they can differ in is
# control flow: a handler that reaches the next one instead of dispatching
# produces wrong answers rather than a crash, and it does so in only one of the
# two builds. That is invisible to a suite run against a single build.
#
# So both are built and every program in tests/ is run through each, comparing
# output and exit status. This is the check that makes the threaded build
# shippable rather than merely passing.

set -uo pipefail
cd "$(dirname "$0")/.."

WORK=$(mktemp -d -t dispatcheq.XXXXXX)
cleanup() {
    status=$?
    if [ "$status" -eq 0 ]; then
        rm -rf "$WORK"
    else
        printf 'I retained dispatch logs in %s\n' "$WORK" >&2
    fi
}
trap cleanup EXIT

FLAGS="-Wall -Wextra -Werror -std=c99 -g -O2 -Isrc -D_GNU_SOURCE"
if [ -d /opt/homebrew/opt/openssl@3/include ]; then
    FLAGS="$FLAGS -I/opt/homebrew/opt/openssl@3/include"
fi

echo "Building both dispatch strategies..."
make nano_virt >/dev/null || { echo "cannot build nano_virt"; exit 1; }

build_vm() {
    local out="$1"; shift
    rm -f obj/nanovm/vm.o
    make nano_vm CFLAGS="$FLAGS $*" >"$out.build.log" 2>&1 || return 1
    cp bin/nano_vm "$out"
}

build_vm "$WORK/vm_goto"   || { echo "threaded build failed"; exit 1; }
build_vm "$WORK/vm_switch" -DNANO_NO_COMPUTED_GOTO \
    || { echo "switch build failed"; exit 1; }
rm -f obj/nanovm/vm.o
make nano_vm >"$WORK/restore.build.log" 2>&1 || exit 1

shopt -s nullglob
sources=(tests/*.nano)
python3 tests/dispatch_equivalence.py --compiler ./bin/nano_virt \
    --vm-goto "$WORK/vm_goto" --vm-switch "$WORK/vm_switch" \
    --logs "$WORK/programs" "${sources[@]}"
