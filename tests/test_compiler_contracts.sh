#!/usr/bin/env bash
# Public compiler contracts for machine-readable output and analysis modes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPILER="${1:-$PROJECT_ROOT/bin/nanoc_c}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/nanolang-compiler-contracts.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

if [ ! -x "$COMPILER" ]; then
    echo "ERROR: compiler not found at $COMPILER" >&2
    exit 1
fi

cat >"$WORK/module.nano" <<'NANO'
struct Point {
    x: int,
    y: int
}

pub fn distance_squared(point: Point) -> int {
    return (+ (* point.x point.x) (* point.y point.y))
}

fn main() -> int {
    let point: Point = Point { x: 3, y: 4 }
    (println (int_to_string (distance_squared point)))
    return 0
}

shadow distance_squared {
    assert (== (distance_squared Point { x: 3, y: 4 }) 25)
}

shadow main {
    assert (== (main) 0)
}
NANO

cat >"$WORK/error.nano" <<'NANO'
fn main() -> int {
    return "not an int"
}
NANO

failures=0
pass() { echo "  ✓ $1"; }
fail() { echo "  ✗ $1" >&2; failures=$((failures + 1)); }

echo "Compiler public-contract tests"

reflect="$WORK/reflect.json"
if "$COMPILER" "$WORK/module.nano" --reflect "$reflect" >/dev/null 2>&1 \
        && python3 - "$reflect" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
text = json.dumps(data)
assert "distance_squared" in text
assert "Point" in text
PY
then
    pass "--reflect writes valid JSON containing public declarations"
else
    fail "--reflect response contract"
fi

typed="$WORK/typed.json"
if "$COMPILER" "$WORK/module.nano" --emit-typed-ast-json >"$typed" 2>/dev/null \
        && python3 - "$typed" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert isinstance(data, dict)
assert "format_version" in data or "functions" in data
PY
then
    pass "--emit-typed-ast-json writes a versioned JSON object"
else
    fail "--emit-typed-ast-json response contract"
fi

diag="$WORK/diagnostics.json"
set +e
"$COMPILER" "$WORK/error.nano" --llm-diags-json "$diag" >/dev/null 2>&1
diag_status=$?
set -e
if [ "$diag_status" -ne 0 ] && python3 - "$diag" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert data.get("success") is False
diagnostics = data.get("diagnostics")
assert isinstance(diagnostics, list) and diagnostics
item = diagnostics[0]
for key in ("severity", "message"):
    assert key in item
PY
then
    pass "--llm-diags-json couples failure status to a populated error schema"
else
    fail "--llm-diags-json error contract"
fi

toon="$WORK/diagnostics.toon"
set +e
"$COMPILER" "$WORK/error.nano" --llm-diags-toon "$toon" >/dev/null 2>&1
toon_status=$?
set -e
if [ "$toon_status" -ne 0 ] \
        && grep -q 'diagnostics\[1\]:' "$toon" \
        && grep -q 'severity.*code.*message.*file.*line.*column' "$toon" \
        && grep -q 'diagnostic_count: 1' "$toon"; then
    pass "--llm-diags-toon couples failure status to its documented schema"
else
    fail "--llm-diags-toon error contract"
fi

success_diag="$WORK/success.json"
if "$COMPILER" "$WORK/module.nano" -o "$WORK/module" \
        --llm-diags-json "$success_diag" >/dev/null 2>&1 \
        && python3 - "$success_diag" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert data.get("success") is True
assert data.get("diagnostics") in ([], None)
PY
then
    pass "--llm-diags-json reports successful compilation without errors"
else
    fail "--llm-diags-json success contract"
fi

success_toon="$WORK/success.toon"
if "$COMPILER" "$WORK/module.nano" -o "$WORK/module-toon" \
        --llm-diags-toon "$success_toon" >/dev/null 2>&1 \
        && grep -q 'diagnostic_count: 0' "$success_toon"; then
    pass "--llm-diags-toon reports successful compilation without errors"
else
    fail "--llm-diags-toon success contract"
fi

trust="$WORK/trust.txt"
if "$COMPILER" "$WORK/module.nano" --trust-report >"$trust" 2>/dev/null \
        && grep -Eiq 'verified|typechecked|nanocore' "$trust"; then
    pass "--trust-report produces trust-level information"
else
    fail "--trust-report output contract"
fi

reference="$WORK/reference.txt"
if "$COMPILER" "$WORK/module.nano" --reference-eval >"$reference" 2>/dev/null \
        && grep -Eiq 'NanoCore Reference|checked|Summary' "$reference"; then
    pass "--reference-eval produces a reference-evaluation report"
else
    fail "--reference-eval output contract"
fi

cat >"$WORK/bench.nano" <<'NANO'
fn bench_add() -> int {
    return (+ 20 22)
}

fn main() -> int {
    return 0
}

shadow bench_add {
    assert (== (bench_add) 42)
}

shadow main {
    assert (== (main) 0)
}
NANO

bench_json="$WORK/bench.json"
if "$COMPILER" "$WORK/bench.nano" --bench --bench-n 2 \
        --bench-json "$bench_json" -o "$WORK/bench" >/dev/null 2>&1 \
        && python3 - "$bench_json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert isinstance(data, (dict, list))
PY
then
    pass "--bench-json writes valid JSON for a fixed iteration count"
else
    fail "--bench-json response contract"
fi

test "$failures" -eq 0
