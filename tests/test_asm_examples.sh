#!/usr/bin/env bash
#
# The symbolic-assembly examples must assemble, verify, and produce their
# documented output.
#
# An example that only looks right is worse than none: it teaches a syntax
# the assembler does not accept, or a stack discipline the verifier rejects.
# Assembling already runs the verifier, so a broken example fails here rather
# than at load time -- and running it checks that the comments describing what
# each word leaves on the stack are true.

set -uo pipefail
cd "$(dirname "$0")/.."

ISA=./bin/nanoisa
VM=./bin/nano_vm
[ -x "$ISA" ] && [ -x "$VM" ] || { echo "need $ISA and $VM built"; exit 1; }

tmp=$(mktemp -t asmex.XXXXXX).nvm
trap 'rm -f "$tmp"' EXIT

fail=0

check() {
    local src="$1"; shift
    local expected="$1"; shift
    printf '  %-42s' "$(basename "$src")"
    if ! "$ISA" asm "$src" -o "$tmp" >/dev/null 2>&1; then
        printf 'FAIL (assemble)\n'
        "$ISA" asm "$src" -o "$tmp" 2>&1 | sed 's/^/       /'
        fail=$((fail + 1)); return
    fi
    local got
    got=$("$VM" "$tmp" 2>&1)
    if [ "$got" != "$expected" ]; then
        printf 'FAIL (output)\n'
        printf '       expected: %s\n' "$(printf '%s' "$expected" | tr '\n' '|')"
        printf '       got:      %s\n' "$(printf '%s' "$got" | tr '\n' '|')"
        fail=$((fail + 1)); return
    fi
    printf 'PASS\n'
}

echo "=== NanoISA symbolic assembly examples ==="

check examples/nanoisa/nanolang_shapes.nasm \
"NanoISA symbolic assembly
fizz"

check examples/nanoisa/forth_words.nasm \
"SQUARE 7, CUBE 3, and 10 gcd 4:
49
27
2"

# Every example must also survive canonical disassembly and reassembly. The
# examples are written by hand with symbolic operands; the disassembler emits
# resolved ones. Both must assemble to the same bytes, or the symbolic form is
# a dialect rather than the same language.
echo
echo "=== round trip through canonical disassembly ==="
for src in examples/nanoisa/*.nasm; do
    printf '  %-42s' "$(basename "$src")"
    if ! "$ISA" asm "$src" -o "$tmp" >/dev/null 2>&1; then
        printf 'SKIP (did not assemble)\n'; continue
    fi
    round=$(mktemp -t asmrt.XXXXXX).nvm
    if "$ISA" dump "$tmp" > "${round}.nasm" 2>/dev/null \
        && "$ISA" asm "${round}.nasm" -o "$round" >/dev/null 2>&1 \
        && cmp -s "$tmp" "$round"; then
        printf 'PASS\n'
    else
        printf 'FAIL\n'; fail=$((fail + 1))
    fi
    rm -f "$round" "${round}.nasm"
done

echo
if [ "$fail" -eq 0 ]; then echo "All assembly example tests passed."; else echo "$fail failed."; fi
[ "$fail" -eq 0 ]
