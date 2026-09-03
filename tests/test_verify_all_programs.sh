#!/usr/bin/env bash
#
# Every program in tests/ must produce bytecode the verifier accepts.
#
# Compiling is not the same as verifying. `make test-quick` runs a subset and
# `examples-core` builds examples, so a program that compiles cleanly and
# then fails verification could sit in tests/ unnoticed -- which is exactly
# what happened when stack-height propagation started reaching past the first
# portable-ISA instruction and found latent codegen bugs in eight files. This
# closes that hole: it verifies every .nano in tests/, and fails on any
# regression against a known-failing allowlist.
#
# The allowlist is for programs whose bytecode the verifier rejects for
# reasons that predate this check. Removing an entry is progress; adding one
# needs a reason in the commit message.

set -uo pipefail
cd "$(dirname "$0")/.."

VIRT=./bin/nano_virt
VM=./bin/nano_vm
[ -x "$VIRT" ] && [ -x "$VM" ] || { echo "need $VIRT and $VM built"; exit 1; }

# Known-failing, each for a reason tracked elsewhere:
#
#   test_import_aliasing.nano  - a function's local_count is smaller than its
#                                arity, so the frame cannot hold its arguments
#
#   The next three contain a transitively-imported function whose float
#   arithmetic lowered to integer opcodes (issue #223). The verifier is right
#   and the bytecode is wrong: the VM traps on I64_ADD with float operands, so
#   these would fail if the affected function were ever called. Removing them
#   from this list is the acceptance test for that fix.
ALLOW="test_import_aliasing.nano test_vector2d.nano contracts_stdlib_test.nano contracts_matrix_timing_test.nano"

tmp=$(mktemp -t verifyall.XXXXXX).nvm
trap 'rm -f "$tmp"' EXIT

checked=0 failed=0 allowed=0 unexpected_pass=0
for f in tests/*.nano; do
    base=$(basename "$f")
    "$VIRT" "$f" -o "$tmp" --emit-nvm >/dev/null 2>&1 || continue
    checked=$((checked + 1))
    msg=$("$VM" "$tmp" 2>&1 | grep -i "verification failed" | head -1)
    listed=0
    case " $ALLOW " in *" $base "*) listed=1;; esac
    if [ -n "$msg" ]; then
        if [ "$listed" = 1 ]; then
            allowed=$((allowed + 1))
        else
            failed=$((failed + 1))
            printf '  FAIL %s\n       %s\n' "$base" "$msg"
        fi
    elif [ "$listed" = 1 ]; then
        unexpected_pass=$((unexpected_pass + 1))
        printf '  NOW PASSES %s -- remove it from ALLOW in %s\n' "$base" "$0"
    fi
done

printf '\n%d programs verified, %d failures, %d known-failing' \
    "$checked" "$failed" "$allowed"
[ "$unexpected_pass" -gt 0 ] && printf ', %d newly passing' "$unexpected_pass"
printf '\n'

[ "$failed" -eq 0 ] && [ "$unexpected_pass" -eq 0 ]
