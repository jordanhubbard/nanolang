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
trap 'rm -rf "$WORK"' EXIT

FLAGS="-Wall -Wextra -Werror -std=c99 -g -O2 -Isrc -D_GNU_SOURCE"
if [ -d /opt/homebrew/opt/openssl@3/include ]; then
    FLAGS="$FLAGS -I/opt/homebrew/opt/openssl@3/include"
fi

echo "Building both dispatch strategies..."
make nano_virt >/dev/null || { echo "cannot build nano_virt"; exit 1; }

build_vm() {
    local out="$1"; shift
    rm -f obj/nanovm/vm.o
    make nano_vm CFLAGS="$FLAGS $*" >/dev/null 2>&1 || return 1
    cp bin/nano_vm "$out"
}

build_vm "$WORK/vm_goto"   || { echo "threaded build failed"; exit 1; }
build_vm "$WORK/vm_switch" -DNANO_NO_COMPUTED_GOTO \
    || { echo "switch build failed"; exit 1; }
rm -f obj/nanovm/vm.o
make nano_vm >/dev/null 2>&1

same=0 differ=0
for src in tests/*.nano; do
    nvm="$WORK/program.nvm"
    ./bin/nano_virt "$src" -o "$nvm" --emit-nvm >/dev/null 2>&1 || continue
    a=$("$WORK/vm_goto" "$nvm" 2>&1; printf 'rc=%d' $?)
    b=$("$WORK/vm_switch" "$nvm" 2>&1; printf 'rc=%d' $?)
    # The VM reports its own argv[0], which necessarily differs.
    a=${a//vm_goto/VM}
    b=${b//vm_switch/VM}
    if [ "$a" = "$b" ]; then
        same=$((same + 1))
    else
        differ=$((differ + 1))
        printf '  DIFFER %s\n' "$(basename "$src")"
        diff <(printf '%s' "$a") <(printf '%s' "$b") | head -6 | sed 's/^/       /'
    fi
done

printf '\n%d programs identical under both dispatch strategies, %d differing\n' \
    "$same" "$differ"
[ "$differ" -eq 0 ]
