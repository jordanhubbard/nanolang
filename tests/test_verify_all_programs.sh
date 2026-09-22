#!/usr/bin/env bash
# I verify every root test source, with explicit failures and retained logs.
set -euo pipefail
shopt -s nullglob
cd "$(dirname "$0")/.."

if [[ -n "${NANO_VERIFY_CORPUS_LOG_ROOT:-}" ]]; then
    mkdir -p "$NANO_VERIFY_CORPUS_LOG_ROOT"
    work=$(mktemp -d "$NANO_VERIFY_CORPUS_LOG_ROOT/corpus.XXXXXX")
else
    work=$(mktemp -d "${TMPDIR:-/tmp}/nano-verify-corpus.XXXXXX")
fi
if python3 tests/verify_corpus.py --compiler ./bin/nano_virt --vm ./bin/nano_vm \
        --logs "$work" tests/*.nano; then
    rm -r "$work"
else
    status=$?
    printf 'I retained verifier logs in %s\n' "$work" >&2
    exit "$status"
fi
