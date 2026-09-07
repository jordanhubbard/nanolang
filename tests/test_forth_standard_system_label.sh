#!/usr/bin/env bash
# Precise Forth 2012 system label. Passing Jackson files is evidence.
# This gate fails if I claim a Standard System or drop the denial.

set -uo pipefail
cd "$(dirname "$0")/.."

DOC=docs/FORTH_STANDARD_SYSTEM.md
PIN=docs/FORTH_2012.md
COV=docs/FORTH_CORE_COVERAGE.md

fail=0
pass() { printf '  PASS  %s\n' "$1"; }
fail_msg() { printf '  FAIL  %s\n' "$1"; fail=$((fail + 1)); }

echo "=== Forth 2012 system label ==="

python3 - "$DOC" "$PIN" "$COV" <<'PY'
import re, sys

doc_path, pin_path, cov_path = sys.argv[1:4]
doc = open(doc_path, encoding="utf-8").read()
pin = open(pin_path, encoding="utf-8").read()
cov = open(cov_path, encoding="utf-8").read()
errors = []

if "I am not a Forth 2012 Standard System" not in doc:
    errors.append("docs/FORTH_STANDARD_SYSTEM.md must deny a Forth 2012 Standard System")
if "I am not an ANS Forth Standard System" not in doc:
    errors.append("docs/FORTH_STANDARD_SYSTEM.md must deny an ANS Forth Standard System")
if "I do not claim Core" not in doc:
    errors.append("docs/FORTH_STANDARD_SYSTEM.md must deny a Core banner")
if re.search(r"(?m)^I am a Forth 2012 Standard System", doc):
    errors.append("docs/FORTH_STANDARD_SYSTEM.md claimed a Standard System")
if "INCLUDE" not in doc and "INCLUDED" not in doc:
    errors.append("docs/FORTH_STANDARD_SYSTEM.md must cite the INCLUDE gap")
if "FORTH_CORE_COVERAGE.md" not in doc:
    errors.append("docs/FORTH_STANDARD_SYSTEM.md must cite the coverage matrix")
if "Until the pinned suites pass" in pin:
    errors.append("docs/FORTH_2012.md still says the pinned suites have not passed")
if "FORTH_STANDARD_SYSTEM.md" not in pin:
    errors.append("docs/FORTH_2012.md must point at the precise label")
if "I am not a Forth 2012 Standard System" not in pin:
    errors.append("docs/FORTH_2012.md dropped the non-conformance sentence")
if "I do not claim Core" not in cov:
    errors.append("docs/FORTH_CORE_COVERAGE.md must still deny Core")

if errors:
    for e in errors:
        print("FAIL", e)
    sys.exit(1)
print("label document denials, INCLUDE gap, coverage matrix, and pin cross-links")
sys.exit(0)
PY
if [ $? -eq 0 ]; then
    pass "label document denials and pin cross-links"
else
    fail_msg "label document denials and pin cross-links"
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "Forth 2012 system label checks passed. I do not claim a Standard System."
    exit 0
fi
echo "$fail Forth 2012 system label check(s) failed."
exit 1
