#!/bin/sh
set -eu

root="${TMPDIR:-/tmp}/nanolang-selfhost-alias-$$"
trap 'rm -rf "$root"' EXIT HUP INT TERM
mkdir -p "$root/deps"

cat >"$root/root.nano" <<'EOF'
import "./deps/value.nano"
fn main() -> int { return (value) }
shadow main { assert true }
EOF
cat >"$root/deps/value.nano" <<'EOF'
module value
pub fn value() -> int { return 0 }
shadow value { assert (== (value) 0) }
EOF

compiler="${NANOC_SELFHOST:-./bin/nanoc_v06}"

expect_rejected() {
    source="$1"
    destination="$2"
    kind="$3"
    before="$(cksum <"$source")"
    if "$compiler" "$root/root.nano" "$kind" "$destination" >"$root/log" 2>&1; then
        echo "expected alias rejection: $kind $destination" >&2
        exit 1
    fi
    test "$(cksum <"$source")" = "$before"
}

expect_rejected "$root/root.nano" "$root/root.nano" -o
expect_rejected "$root/root.nano" "$root/./root.nano" -o
ln -s root.nano "$root/root-link.nano"
expect_rejected "$root/root.nano" "$root/root-link.nano" -o
ln "$root/root.nano" "$root/root-hard.nano"
expect_rejected "$root/root.nano" "$root/root-hard.nano" -o

cat >"$root/broken.nano" <<'EOF'
fn main() -> int { return "wrong" }
shadow main { assert true }
EOF
ln "$root/broken.nano" "$root/broken-diags.json"
before="$(cksum <"$root/broken.nano")"
if "$compiler" "$root/broken.nano" --llm-diags-json "$root/broken-diags.json" >"$root/log" 2>&1; then
    echo "expected diagnostic alias rejection" >&2
    exit 1
fi
test "$(cksum <"$root/broken.nano")" = "$before"

expect_rejected "$root/deps/value.nano" "$root/deps/value.nano" -o
ln -s value.nano "$root/deps/value-link.nano"
expect_rejected "$root/deps/value.nano" "$root/deps/value-link.nano" -o
ln "$root/deps/value.nano" "$root/deps/value-hard.nano"
expect_rejected "$root/deps/value.nano" "$root/deps/value-hard.nano" -o

expect_rejected "$root/root.nano" "$root/root.nano" --llm-diags-json
expect_rejected "$root/deps/value.nano" "$root/deps/value.nano" --llm-diags-json

ln -s loop "$root/loop"
if "$compiler" "$root/root.nano" -o "$root/loop" >"$root/log" 2>&1; then
    echo "expected identity lookup failure" >&2
    exit 1
fi

echo "self-hosted source/destination identity checks passed"
