#!/usr/bin/env bash
# check_formal_claims.sh — keep the formal-verification docs honest.
#
# The formal/README.md previously advertised "0 Admitted / axiom-free" while
# Equivalence.v carried an `Admitted`. Documentation claims about proof
# completeness must not drift from the actual Coq sources, so this gate compares
# the real counts against what the README asserts and fails on any mismatch.
#
# Run from the repo root (CI + `make test`).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

FORMAL_DIR="formal"
README="$FORMAL_DIR/README.md"

if [ ! -d "$FORMAL_DIR" ]; then
    echo "check_formal_claims: no $FORMAL_DIR directory; skipping"
    exit 0
fi

# Number of incomplete proofs = number of `Admitted.` proof-closers. An `admit.`
# tactic always ends its proof in `Admitted`, so counting the closer avoids
# double-counting an `admit`/`Admitted` pair as two incomplete proofs.
# `|| true`: grep exits non-zero when there are no matches, which (under
# `set -e` + `pipefail`) would otherwise abort the script — a no-match is the
# healthy case here, not an error.
actual_admitted=$( { grep -rEho '\bAdmitted\b' "$FORMAL_DIR"/*.v 2>/dev/null || true; } | wc -l | tr -d ' ')
# Separately, presence of any `admit`/`Admitted` token means "not fully closed".
actual_incomplete_tokens=$( { grep -rEho '\bAdmitted\b|\badmit\b' "$FORMAL_DIR"/*.v 2>/dev/null || true; } | wc -l | tr -d ' ')
actual_axioms=$( { grep -rEho '^[[:space:]]*Axiom\b' "$FORMAL_DIR"/*.v 2>/dev/null || true; } | wc -l | tr -d ' ')

echo "check_formal_claims: actual Admitted/admit=$actual_admitted, Axiom=$actual_axioms"

fail=0

# If the README claims a specific Admitted count, it must match reality.
claimed_admitted=$(grep -oE '\*\*Admitted:\*\*[[:space:]]*[0-9]+' "$README" 2>/dev/null \
    | grep -oE '[0-9]+' | head -1 || true)
if [ -n "${claimed_admitted:-}" ] && [ "$claimed_admitted" != "$actual_admitted" ]; then
    echo "FAIL: $README claims Admitted: $claimed_admitted but sources have $actual_admitted" >&2
    fail=1
fi

# The README must not assert "0 Admitted" (in prose) when there are any.
if [ "$actual_incomplete_tokens" -gt 0 ] && grep -qiE '0[[:space:]]*`?Admitted' "$README" 2>/dev/null; then
    echo "FAIL: $README asserts '0 Admitted' but sources have $actual_admitted" >&2
    fail=1
fi

# "axiom-free" is only honest when there are genuinely no Axiom declarations.
if [ "$actual_axioms" -gt 0 ] && grep -qiE 'axiom-free' "$README" 2>/dev/null; then
    echo "FAIL: $README claims 'axiom-free' but sources declare $actual_axioms axiom(s)" >&2
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    echo "check_formal_claims: FAILED — update $README to match the Coq sources." >&2
    exit 1
fi

echo "check_formal_claims: OK — formal docs match the proof state."
