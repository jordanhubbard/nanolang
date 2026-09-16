#!/usr/bin/env bash
# I verify every root test source, with explicit failures and retained logs.
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."

work=$(mktemp -d "${TMPDIR:-/tmp}/nano-verify-corpus.XXXXXX")
if python3 tests/verify_corpus.py --compiler ./bin/nano_virt --vm ./bin/nano_vm \
        --logs "$work" tests/*.nano; then
    rm -r "$work"
else
    status=$?
    printf 'I retained verifier logs in %s\n' "$work" >&2
    exit "$status"
fi
