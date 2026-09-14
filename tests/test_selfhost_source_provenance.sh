#!/bin/sh
set -eu

root="$PWD/tests/.provenance-\"quoted-$$"
trap 'rm -rf "$root"' EXIT HUP INT TERM
mkdir -p "$root/nested"

cat >"$root/root.nano" <<'EOF'
import "./nested/middle.nano"
fn main() -> int {
    return (broken 1)
}
shadow main { assert true }
EOF

cat >"$root/nested/middle.nano" <<'EOF'
module middle
import "./broken.nano"
EOF

cat >"$root/nested/broken.nano" <<'EOF'
module broken

pub fn broken(value: int) -> int {
    return "wrong"
}
shadow broken { assert true }
EOF

json="$root/diagnostics.json"
compiler="${NANOC_SELFHOST:-./bin/nanoc_v06}"
if "$compiler" "$root/root.nano" -o "$root/out" --llm-diags-json "$json" >"$root/output" 2>&1; then
    echo "expected compilation to fail" >&2
    exit 1
fi

python3 - "$json" "$root/nested/broken.nano" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    result = json.load(stream)
locations = [item["location"] for item in result["diagnostics"]]
assert any(loc["file"] == sys.argv[2] and loc["line"] == 4 for loc in locations), locations
assert all("nanolang_merged.nano" not in loc["file"] for loc in locations), locations
PY
