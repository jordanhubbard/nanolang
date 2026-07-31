#!/usr/bin/env bash
# tests/check_stdlib_docs.sh — keep docs/STDLIB.md in sync with the builtin registry
#
# src/builtins_registry.c is the single authoritative table of every builtin
# function the language exposes. docs/STDLIB.md is the hand-written reference
# for those same functions, so the two drift apart silently every time a
# builtin is added or removed. This checker makes that drift mechanical:
#
#   1. it extracts the builtin names from the registry table in src/,
#   2. it extracts the documented names from the `### `name(...)`` headings
#      in docs/STDLIB.md,
#   3. it prints both directions of the difference — undocumented builtins
#      and documented functions that no longer exist,
#   4. it verifies each `## Section (N)` heading's declared count matches the
#      number of entries under it, so the section totals cannot rot either.
#
# Usage:
#   tests/check_stdlib_docs.sh              # check for drift
#   tests/check_stdlib_docs.sh --list       # print the registry names, one per line
#   tests/check_stdlib_docs.sh --documented # print the documented names, one per line
#
# Exits 0 when the tree is clean, 1 on drift, 2 on a missing input file.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REGISTRY="${REGISTRY:-$REPO_ROOT/src/builtins_registry.c}"
STDLIB_DOC="${STDLIB_DOC:-$REPO_ROOT/docs/STDLIB.md}"

for f in "$REGISTRY" "$STDLIB_DOC"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: $f not found" >&2
        exit 2
    fi
done

# Names in the registry table. Only lines inside `builtin_registry[] = { ... }`
# that begin an entry (`{"name", ...`) count; comments and the lookup helpers
# below the table are skipped.
list_registry() {
    awk '
        /^const BuiltinEntry builtin_registry\[\] = \{/ { in_table = 1; next }
        in_table && /^\};/                             { in_table = 0; next }
        in_table && match($0, /^[ \t]*\{[ \t]*"[A-Za-z_][A-Za-z0-9_]*"/) {
            entry = substr($0, RSTART, RLENGTH)
            sub(/^[ \t]*\{[ \t]*"/, "", entry)
            sub(/"$/, "", entry)
            print entry
        }
    ' "$REGISTRY" | sort -u
}

# Names documented as `### `name(args) -> type`` headings.
list_documented() {
    sed -n 's/^### `\([A-Za-z_][A-Za-z0-9_]*\)(.*/\1/p' "$STDLIB_DOC" | sort -u
}

case "${1:-}" in
    --list|--list-registry)  list_registry;  exit 0 ;;
    --documented|--list-documented) list_documented; exit 0 ;;
    "") ;;
    *) echo "ERROR: unknown option '$1' (see --help in the file header)" >&2; exit 2 ;;
esac

TMPDIR_BASE="${TMPDIR:-/tmp}"
REG_LIST="$TMPDIR_BASE/check_stdlib_docs_reg_$$"
DOC_LIST="$TMPDIR_BASE/check_stdlib_docs_doc_$$"
trap 'rm -f "$REG_LIST" "$DOC_LIST"' EXIT

list_registry   > "$REG_LIST"
list_documented > "$DOC_LIST"

reg_count=$(wc -l < "$REG_LIST" | tr -d ' ')
doc_count=$(wc -l < "$DOC_LIST" | tr -d ' ')

if [ "$reg_count" -eq 0 ]; then
    echo "ERROR: no builtins extracted from $REGISTRY — has the table format changed?" >&2
    exit 2
fi

undocumented=$(comm -23 "$REG_LIST" "$DOC_LIST")
stale=$(comm -13 "$REG_LIST" "$DOC_LIST")

# Section headings declare their own entry count, e.g. `## Math (20)`.
count_mismatches=$(awk '
    function flush() {
        if (section != "" && declared >= 0 && declared != entries)
            printf "  %s: heading says %d, found %d\n", section, declared, entries
    }
    /^## / {
        flush()
        section = substr($0, 4)
        entries = 0
        declared = -1
        if (match(section, /\([0-9]+\)[ \t]*$/)) {
            declared = substr(section, RSTART + 1, RLENGTH - 2) + 0
        }
        next
    }
    /^### `[A-Za-z_][A-Za-z0-9_]*\(/ { entries++ }
    END { flush() }
' "$STDLIB_DOC")

undocumented_n=$([ -n "$undocumented" ] && echo "$undocumented" | wc -l | tr -d ' ' || echo 0)
stale_n=$([ -n "$stale" ] && echo "$stale" | wc -l | tr -d ' ' || echo 0)

echo "stdlib doc coverage: $reg_count builtins in src/builtins_registry.c, $doc_count documented in docs/STDLIB.md"
echo ""

status=0

if [ -n "$undocumented" ]; then
    status=1
    echo "Undocumented builtins ($undocumented_n) — in the registry, missing from docs/STDLIB.md:"
    echo "$undocumented" | sed 's/^/  - /'
    echo ""
fi

if [ -n "$stale" ]; then
    status=1
    echo "Stale documentation ($stale_n) — documented in docs/STDLIB.md, not in the registry:"
    echo "$stale" | sed 's/^/  - /'
    echo ""
fi

if [ -n "$count_mismatches" ]; then
    status=1
    echo "Section count mismatches — the (N) in the heading disagrees with the entries below it:"
    echo "$count_mismatches"
    echo ""
fi

if [ "$status" -ne 0 ]; then
    echo "FAIL: docs/STDLIB.md has drifted from src/builtins_registry.c."
    echo "      Add a '### \`name(args) -> type\`' section for each undocumented builtin,"
    echo "      remove sections for builtins that no longer exist, and update the"
    echo "      '(N)' count in the affected '## Section' headings."
    exit 1
fi

echo "OK: every builtin is documented and every section count is accurate."
exit 0
