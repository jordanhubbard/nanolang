#!/usr/bin/env bash
# tests/audit_warnings.sh — find generated-C warnings across the example set
#
# Reproduces stage1's cc command for each example, then re-runs it with
# -Wall -Wextra (stage1 itself uses -std=c99 with no -W flags). Any warning
# that appears here is something stage1 silently passes through. See beads
# nl-bar (discarded-qualifiers — historically), nl-pke (int-conversion),
# nl-v77 (HashMap pointer-types — historically) for the surfaces this is
# meant to catch.
#
# Usage:
#   tests/audit_warnings.sh            # audit a curated representative set
#   tests/audit_warnings.sh --all      # audit every .nano under examples/
#   tests/audit_warnings.sh path/to/x.nano [...]  # audit specific files
#
# Exits 0 on no warnings, 1 if any warnings found. Prints summary by type.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
# Use the C reference compiler (nanoc_c). Stage1 may parse-error on
# patterns it doesn't yet support; nanoc_c covers the full language.
NANOC="${NANOC:-$REPO_ROOT/bin/nanoc_c}"

if [ ! -x "$NANOC" ]; then
    echo "ERROR: nanoc_c not found at $NANOC. Run 'make build' first." >&2
    exit 2
fi

if [ "${1:-}" = "--all" ]; then
    targets=$(find "$REPO_ROOT/examples" -name "*.nano" 2>/dev/null)
elif [ $# -gt 0 ]; then
    targets="$*"
else
    # Curated representative set
    targets=$(cat <<'EOF'
examples/language/nl_fibonacci.nano
examples/language/nl_hashmap.nano
examples/language/nl_match_int.nano
examples/language/nl_string_operations.nano
examples/language/nl_struct.nano
examples/language/nl_union_types.nano
examples/language/nl_types_union_construct.nano
examples/language/nl_array_complete.nano
examples/language/nl_for_in_array.nano
examples/language/nl_generics_demo.nano
examples/opl/opl_cli.nano
EOF
)
    targets=$(cd "$REPO_ROOT" && for f in $targets; do [ -f "$f" ] && echo "$REPO_ROOT/$f"; done)
fi

LOG="${TMPDIR:-/tmp}/audit_warnings_$$.log"
> "$LOG"
trap 'rm -f "$LOG"' EXIT

audited=0
clean=0
failed=0
for f in $targets; do
    [ -f "$f" ] || continue
    audited=$((audited+1))
    out=$("$NANOC" "$f" -o /tmp/audit_bin 2>&1)
    ec=$?
    # Strip nanoc-internal Warning lines (shadow tests etc.); keep cc warnings/errors
    cc_diags=$(echo "$out" | grep -E "warning:|error:" | grep -v "^Warning:" | grep -v "^Error:")
    if [ $ec -eq 0 ] && [ -z "$cc_diags" ]; then
        clean=$((clean+1))
    else
        failed=$((failed+1))
        echo "FAIL: $f" >> "$LOG"
        echo "$cc_diags" >> "$LOG"
        echo "---" >> "$LOG"
    fi
    rm -f /tmp/audit_bin
done

echo "Audited $audited examples: $clean clean, $failed with cc-level diagnostics"
echo ""
if [ "$failed" -gt 0 ]; then
    echo "Diagnostic types:"
    grep -oE "\[-W[a-z-]+\]" "$LOG" | sort | uniq -c | sort -rn || true
    echo ""
    echo "By file (top 10):"
    grep -E "^FAIL: " "$LOG" | sort -u | head -10
    echo ""
    echo "First 5 sample diagnostics:"
    grep -E "warning:|error:" "$LOG" | head -5
    exit 1
fi
exit 0
