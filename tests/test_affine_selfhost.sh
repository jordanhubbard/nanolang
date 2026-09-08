#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
tmp=${TMPDIR:-/tmp}/nanolang-affine-selfhost.$$
trap 'rm -rf "$tmp"' EXIT
mkdir -p "$tmp"

check_accept() {
    local compiler=$1
    local source=$2
    "$compiler" "$source" -o "$tmp/out"
}

check_reject() {
    local compiler=$1
    local source=$2
    if "$compiler" "$source" -o "$tmp/out" >"$tmp/stdout" 2>"$tmp/stderr"; then
        printf 'expected rejection: %s %s\n' "$compiler" "$source" >&2
        return 1
    fi
}

check_accept "$root/bin/nanoc_c" "$root/tests/affine_selfhost/valid_move.nano"

for compiler in "$root/bin/nanoc_stage1" "$root/bin/nanoc_stage2"; do
    check_accept "$compiler" "$root/tests/affine_selfhost/valid_move.nano"
    check_reject "$compiler" "$root/tests/affine_selfhost/use_after_move.nano"
    check_reject "$compiler" "$root/tests/affine_selfhost/unresolved.nano"
done

printf 'affine self-host parity: PASS\n'
