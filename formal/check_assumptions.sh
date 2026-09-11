#!/usr/bin/env bash
# I fail closed on missing reports, axioms, or an unexpected report format.
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo 'Usage: check_assumptions.sh SOURCE REPORT [REQUIRED_THEOREM ...]' >&2
    exit 2
fi
source_file=$1
report_file=$2
shift 2
if [[ $# == 0 ]]; then
    set -- preservation_contract progress_contract determinism_contract equivalence_contract evaluator_contract
fi
for theorem in "$@"; do
    if ! awk -v name="$theorem" '$0 == "Print Assumptions " name "." { found=1 } END { exit !found }' "$source_file"; then
        echo "I require an assumption report for $theorem." >&2
        exit 1
    fi
done
expected=$(awk '/^Print Assumptions [A-Za-z_][A-Za-z_0-9.]*\.$/ { n++ } END { print n+0 }' "$source_file")
if [[ "$expected" == 0 ]]; then
    echo 'I found no named assumption checks.' >&2
    exit 1
fi
awk -v expected="$expected" '
    /^[[:space:]]*$/ { next }
    /^Closed under the global context$/ { closed++; next }
    { print "I reject this assumption report line: " $0 > "/dev/stderr"; bad=1 }
    END {
        if (closed != expected) {
            print "I expected " expected " closed reports; found " closed+0 "." > "/dev/stderr"
            bad=1
        }
        exit bad ? 1 : 0
    }
' "$report_file"
