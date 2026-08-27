#!/usr/bin/env bash
# Public compiler contract for malformed programs.
#
# Every fixture must be rejected, must produce a diagnostic, and must not
# leave an executable behind. High-value namespace regressions additionally
# pin their documented error or warning text.

set -uo pipefail
shopt -s globstar nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPILER="${NANOC:-$PROJECT_ROOT/bin/nanoc}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/nanolang-negative.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

if [ ! -x "$COMPILER" ]; then
    echo "ERROR: compiler not found at $COMPILER" >&2
    exit 1
fi

passed=0
failed=0

expected_diagnostic() {
    case "$1" in
        */duplicate_functions/duplicate_function.nano)
            echo "Function 'add' is already defined" ;;
        */builtin_collision/redefine_abs.nano)
            echo "Cannot redefine built-in function 'abs'" ;;
        */builtin_collision/redefine_min.nano)
            echo "Cannot redefine built-in function 'min'" ;;
        *) echo "error|Error|ERROR|failed|FAILED|UNDEFINED|Undefined" ;;
    esac
}

for source in "$SCRIPT_DIR"/negative/**/*.nano; do
    relative="${source#"$SCRIPT_DIR"/}"
    output="$WORK/${relative//\//_}.out"
    log="$WORK/${relative//\//_}.log"
    pattern="$(expected_diagnostic "$source")"

    printf "%-65s " "$relative"
    if perl -e 'alarm 60; exec @ARGV' \
            "$COMPILER" "$source" -o "$output" >"$log" 2>&1; then
        echo "FAIL (compiler accepted invalid input)"
        failed=$((failed + 1))
        continue
    fi

    if [ -e "$output" ]; then
        echo "FAIL (failed compile left an output artifact)"
        failed=$((failed + 1))
        continue
    fi

    if ! grep -Eq "$pattern" "$log"; then
        echo "FAIL (missing expected diagnostic: $pattern)"
        tail -10 "$log"
        failed=$((failed + 1))
        continue
    fi

    echo "PASS"
    passed=$((passed + 1))
done

total=$((passed + failed))
echo "Negative compiler contracts: $passed/$total passed"
test "$failed" -eq 0

