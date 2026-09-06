#!/usr/bin/env bash
# Pin consistency for Forth 2012 revisions, plus Gforth differential runs of
# examples/language/forth/pi.fs. The pins live in tests/forth/pins.json and
# must appear in docs/FORTH_2012.md. I do not claim a Forth 2012 system.

set -uo pipefail
cd "$(dirname "$0")/.."

PINS=tests/forth/pins.json
DOC=docs/FORTH_2012.md
PI=examples/language/forth/pi.fs

fail=0
pass() { printf '  PASS  %s\n' "$1"; }
fail_msg() { printf '  FAIL  %s\n' "$1"; fail=$((fail + 1)); }

echo "=== Forth 2012 pin consistency ==="

if [ ! -f "$PINS" ]; then
    echo "missing $PINS"
    exit 1
fi
if [ ! -f "$DOC" ]; then
    echo "missing $DOC"
    exit 1
fi
if [ ! -f "$PI" ]; then
    echo "missing $PI"
    exit 1
fi

python3 - "$PINS" "$DOC" <<'PY'
import json, sys

pins_path, doc_path = sys.argv[1], sys.argv[2]
pins = json.load(open(pins_path))
doc = open(doc_path).read()
errors = []

if pins.get("schema") != "nanolang.forth_pins.v1":
    errors.append("pins.json schema is not nanolang.forth_pins.v1")

needles = [
    pins["standard"]["name"],
    pins["test_suites"]["gerryjackson"]["tag"],
    pins["test_suites"]["gerryjackson"]["sha"],
    pins["test_suites"]["forth200x"]["sha"],
    pins["gforth"]["version"],
]
for needle in needles:
    if needle not in doc:
        errors.append("docs/FORTH_2012.md missing pin %r" % needle)

if "I am not a Forth 2012 Standard System" not in doc:
    errors.append("docs/FORTH_2012.md dropped the non-conformance sentence")

if pins["test_suites"]["gerryjackson"].get("vendor") is not True:
    errors.append("gerryjackson.vendor must be true; the suite is vendored")
if pins["test_suites"]["forth200x"].get("vendor") is not False:
    errors.append("forth200x.vendor must stay false; see docs/FORTH_200X_INVENTORY.md")
if pins["gforth"].get("vendor") is not False:
    errors.append("gforth.vendor must stay false; do not vendor GPL Gforth")

for case in pins["pi_cases"]:
    if case["output"] not in doc:
        errors.append("docs/FORTH_2012.md missing pi output for %s places" % case["places"])

if errors:
    for e in errors:
        print("DOCERR", e)
    sys.exit(1)
print("DOCOK", len(needles), "needles")
PY
if [ "$?" -ne 0 ]; then
    fail_msg "docs/FORTH_2012.md drifted from tests/forth/pins.json"
else
    pass "docs/FORTH_2012.md matches tests/forth/pins.json"
fi

echo
echo "=== Gforth differential (pi.fs) ==="

GFORTH_VERSION=$(python3 -c "import json; print(json.load(open('$PINS'))['gforth']['version'])")

if ! command -v gforth >/dev/null 2>&1; then
    if [ -n "${CI:-}" ]; then
        fail_msg "gforth is missing in CI; install the pinned Gforth $GFORTH_VERSION"
    else
        echo "  SKIP  gforth not installed (pin is $GFORTH_VERSION); pin checks already ran"
    fi
else
    got_ver=$(gforth --version 2>&1 | awk '{print $2; exit}')
    if [ "$got_ver" != "$GFORTH_VERSION" ]; then
        fail_msg "gforth version is $got_ver, pin is $GFORTH_VERSION"
    else
        pass "gforth $got_ver matches pin"

        while IFS= read -r line; do
            places=${line%%$'\t'*}
            expected=${line#*$'\t'}
            output=$(gforth -e 'warnings off' "$PI" -e "$places PI bye" 2>/dev/null | tr -d '\r')
            output=${output%$'\n'}
            if [ "$output" = "$expected" ]; then
                pass "pi.fs $places places"
            else
                fail_msg "pi.fs $places places: expected '$expected' got '$output'"
            fi
        done < <(python3 -c "import json; pins=json.load(open('$PINS'));
[print('%s\t%s' % (c['places'], c['output'])) for c in pins['pi_cases']]")
    fi
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "Forth 2012 pin and Gforth differential checks passed."
    exit 0
fi
echo "$fail Forth 2012 pin/Gforth check(s) failed."
exit 1
