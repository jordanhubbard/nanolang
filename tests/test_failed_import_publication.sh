#!/usr/bin/env bash

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIRT="$ROOT/bin/nano_virt"
WORK="$(mktemp -d "$ROOT/.tmp_failed_import.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

failures=0

check_rejected() {
    local name="$1"
    local source="$2"
    local output="$WORK/$name.nvm"
    local before after status

    printf 'previous bytecode' >"$output"
    before="$(cksum "$output")"
    "$VIRT" "$source" --emit-nvm -o "$output" >"$WORK/$name.log" 2>&1
    status=$?
    after="$(cksum "$output")"

    if [ "$status" -eq 0 ]; then
        printf 'FAIL: %s import compiled successfully\n' "$name"
        failures=$((failures + 1))
    elif [ "$before" != "$after" ]; then
        printf 'FAIL: %s import changed the previous output\n' "$name"
        failures=$((failures + 1))
    else
        printf 'PASS: %s import failed without publishing output\n' "$name"
    fi
}

cat >"$WORK/duplicate.nano" <<'EOF'
fn repeated() -> int {
    return 1
}

shadow repeated {
    assert (== (repeated) 1)
}

fn repeated() -> int {
    return 2
}

shadow repeated {
    assert (== (repeated) 2)
}
EOF

cat >"$WORK/direct.nano" <<'EOF'
module "duplicate.nano" as duplicate

fn main() -> int {
    return 0
}

shadow main {
    assert (== (main) 0)
}
EOF

cat >"$WORK/middle.nano" <<'EOF'
module "duplicate.nano" as duplicate
EOF

cat >"$WORK/transitive.nano" <<'EOF'
module "middle.nano" as middle

fn main() -> int {
    return 0
}

shadow main {
    assert (== (main) 0)
}
EOF


check_rejected direct "$WORK/direct.nano"
check_rejected transitive "$WORK/transitive.nano"

if [ "$failures" -ne 0 ]; then
    exit 1
fi
