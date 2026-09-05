#!/usr/bin/env bash
# Jackson v0.15.0 vendor pin, Core-evidence classification, and the precise
# INCLUDE/file-access gap. I do not run optional word sets. I do not claim Core.

set -uo pipefail
cd "$(dirname "$0")/.."

PINS=tests/forth/pins.json
DOC=docs/FORTH_2012.md
COV=docs/FORTH_CORE_COVERAGE.md
INV=docs/FORTH_200X_INVENTORY.md
VENDOR=tests/forth/vendor/gerryjackson
CORE_FR=$VENDOR/src/core.fr
EVIDENCE=tests/forth/core_evidence.txt
WORDS=tests/forth/forth2012_core_words.txt
PROBE=tests/forth/test_forth_include_gap
FORTH=bin/forth
NANO_FORTH=bin/nano_forth

fail=0
pass() { printf '  PASS  %s\n' "$1"; }
fail_msg() { printf '  FAIL  %s\n' "$1"; fail=$((fail + 1)); }

echo "=== Jackson vendor pin ==="

python3 - "$PINS" "$DOC" "$INV" "$VENDOR" "$CORE_FR" <<'PY'
import json, os, sys

pins_path, doc_path, inv_path, vendor, core_fr = sys.argv[1:6]
pins = json.load(open(pins_path))
doc = open(doc_path).read()
errors = []

gj = pins["test_suites"]["gerryjackson"]
if gj.get("vendor") is not True:
    errors.append("gerryjackson.vendor must be true; the suite is vendored")
if gj.get("tag") != "v0.15.0":
    errors.append("gerryjackson tag drifted from v0.15.0")
if gj.get("sha") != "9773f84dd12390f342d37195da8848b04e1f4a23":
    errors.append("gerryjackson sha drifted from 9773f84dd12390f342d37195da8848b04e1f4a23")
if pins["test_suites"]["forth200x"].get("vendor") is not False:
    errors.append("forth200x.vendor must stay false; I do not vendor that tree")
if pins["gforth"].get("vendor") is not False:
    errors.append("gforth.vendor must stay false")

if not os.path.isfile(core_fr):
    errors.append("missing %s" % core_fr)
else:
    text = open(core_fr, encoding="utf-8", errors="replace").read()
    if "JOHNS HOPKINS UNIVERSITY" not in text:
        errors.append("core.fr lost the Johns Hopkins notice")
    want = "9beba157c1929f2908199b2b13a5c2a349df2d3dcc79a293e13080efcc860f54"
    import hashlib
    got = hashlib.sha256(open(core_fr, "rb").read()).hexdigest()
    if got != want:
        errors.append("core.fr sha256 is %s, pin digest is %s" % (got, want))

if not os.path.isfile(os.path.join(vendor, "src", "runtests.fth")):
    errors.append("vendor snapshot is incomplete; runtests.fth missing")
if not os.path.isfile(os.path.join(vendor, "src", "coreexttest.fth")):
    errors.append("vendor snapshot dropped optional files; pin the full tag")

if gj["sha"] not in doc:
    errors.append("docs/FORTH_2012.md missing jackson sha")
if "v0.15.0" not in doc:
    errors.append("docs/FORTH_2012.md missing jackson tag")
if "There is no NanoISA Forth" in doc:
    errors.append("docs/FORTH_2012.md still says there is no NanoISA Forth")
if "I am not a Forth 2012 Standard System" not in doc:
    errors.append("docs/FORTH_2012.md dropped the non-conformance sentence")
if "I do not claim Core" not in doc:
    errors.append("docs/FORTH_2012.md must say I do not claim Core")
if "INCLUDE and INCLUDED are not Forth words" not in doc:
    errors.append("docs/FORTH_2012.md missing the INCLUDE gap sentence")
if "forth_file_open" not in doc or "forth_source_push_file" not in doc:
    errors.append("docs/FORTH_2012.md missing C file-source helpers in the gap")
if "Standard System" in doc and "I am not a Forth 2012 Standard System" not in doc:
    errors.append("docs/FORTH_2012.md Standard System wording drifted")

if not os.path.isfile(inv_path):
    errors.append("missing %s" % inv_path)
else:
    inv = open(inv_path).read()
    if "I do not vendor" not in inv:
        errors.append("inventory must say I do not vendor forth200x")
    if pins["test_suites"]["forth200x"]["sha"] not in inv:
        errors.append("inventory missing forth200x sha")

if errors:
    for e in errors:
        print("DOCERR", e)
    sys.exit(1)
print("DOCOK")
PY
if [ "$?" -ne 0 ]; then
    fail_msg "Jackson pin / docs / vendor tree"
else
    pass "Jackson pin, notices, and docs"
fi

echo
echo "=== Core evidence vs optional files ==="
python3 - "$VENDOR" "$EVIDENCE" <<'PY'
import os, sys
vendor, evidence_path = sys.argv[1], sys.argv[2]
errors = []
core = []
for line in open(evidence_path):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    core.append(line)
    path = os.path.join(vendor, line)
    if not os.path.isfile(path):
        errors.append("missing Core evidence file %s" % line)

optional = [
    "src/coreexttest.fth",
    "src/blocktest.fth",
    "src/doubletest.fth",
    "src/exceptiontest.fth",
    "src/facilitytest.fth",
    "src/filetest.fth",
    "src/localstest.fth",
    "src/memorytest.fth",
    "src/toolstest.fth",
    "src/searchordertest.fth",
    "src/stringtest.fth",
    "src/runtests.fth",
]
for rel in optional:
    if rel in core:
        errors.append("%s must not be Core evidence" % rel)
    if not os.path.isfile(os.path.join(vendor, rel)):
        errors.append("optional file missing from full pin: %s" % rel)

if errors:
    for e in errors:
        print("DOCERR", e)
    sys.exit(1)
print("DOCOK", len(core), "core evidence files")
PY
if [ "$?" -ne 0 ]; then
    fail_msg "Core evidence classification"
else
    pass "optional word-set files are not Core evidence"
fi

echo
echo "=== INCLUDE / INCLUDED gap (NanoISA session) ==="
if [ ! -x "$PROBE" ]; then
    fail_msg "missing $PROBE (make test-forth-jackson must build it)"
else
    find_out=$(perl -e 'alarm 30; exec @ARGV' "$PROBE" "$WORDS" 2>&1)
    probe_rc=$?
    printf '%s\n' "$find_out"
    if [ "$probe_rc" -ne 0 ]; then
        fail_msg "INCLUDE-gap probe"
    else
        pass "INCLUDE-gap probe"
    fi
    python3 - "$COV" "$WORDS" "$find_out" <<'PY'
import sys
cov_path, words_path, find_out = sys.argv[1], sys.argv[2], sys.argv[3]
errors = []
core = [ln.strip() for ln in open(words_path) if ln.strip()]
cov = open(cov_path).read()
if "This is evidence, not a Core pass" not in cov:
    errors.append("coverage matrix must say it is not a Core pass")
if "I do not claim Core" not in cov:
    errors.append("coverage matrix must say I do not claim Core")
for word in core:
    if word not in cov:
        errors.append("coverage missing Core word %r" % word)

status = {}
in_find = False
for line in find_out.splitlines():
    if line.strip() == "COREFIND":
        in_find = True
        continue
    if not in_find:
        continue
    parts = line.split(None, 1)
    if len(parts) != 2 or parts[0] not in ("PRESENT", "MISSING"):
        continue
    status[parts[1]] = parts[0]

for word in core:
    st = status.get(word)
    if st is None:
        errors.append("probe did not report %r" % word)
        continue
    # A table row looks like: | `WORD` | tested | ...
    marker_tested = "| `%s` | tested |" % word
    marker_missing = "| `%s` | missing |" % word
    marker_ambiguous = "| `%s` | ambiguous |" % word
    if st == "MISSING":
        if marker_missing not in cov:
            errors.append("%r is not in the dictionary; coverage must say missing" % word)
        if marker_tested in cov or marker_ambiguous in cov:
            errors.append("%r FIND failed but coverage is not missing" % word)
    else:
        if marker_missing in cov:
            errors.append("%r FIND succeeded; coverage must not say missing" % word)
        if marker_tested not in cov and marker_ambiguous not in cov:
            errors.append("%r FIND succeeded; coverage must be tested or ambiguous" % word)

if errors:
    for e in errors:
        print("DOCERR", e)
    sys.exit(1)
print("DOCOK coverage matches FIND")
PY
    if [ "$?" -ne 0 ]; then
        fail_msg "coverage matrix vs FIND"
    else
        pass "coverage matrix matches FIND"
    fi
fi

echo
echo "=== REPL does not load files; I do not run runtests.fth ==="
if [ ! -x "$FORTH" ] || [ ! -x "$NANO_FORTH" ]; then
    fail_msg "bin/forth and bin/nano_forth must exist"
else
    if [ "$(cmp -s "$FORTH" "$NANO_FORTH"; echo $?)" -ne 0 ]; then
        fail_msg "bin/forth must be a copy of bin/nano_forth"
    else
        pass "bin/forth is bin/nano_forth"
    fi
    repl_out=$(printf 'INCLUDED\nBYE\n' | perl -e 'alarm 15; exec @ARGV' "$FORTH" --interactive 2>&1 || true)
    if printf '%s' "$repl_out" | grep -q 'Forth 2012 Standard System'; then
        fail_msg "REPL claimed a Standard System"
    fi
    if printf '%s' "$repl_out" | grep -q ' ?'; then
        pass "interactive INCLUDED is unknown"
    else
        fail_msg "interactive INCLUDED did not print ?"
        printf '%s\n' "$repl_out"
    fi
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "Jackson vendor, INCLUDE gap, and coverage checks passed. I do not claim Core."
    exit 0
fi
echo "$fail Jackson Core-suite check(s) failed."
exit 1
