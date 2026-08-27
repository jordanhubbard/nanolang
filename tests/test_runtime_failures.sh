#!/usr/bin/env bash
# Programs in this suite must compile, then fail at a documented runtime
# boundary. Compile-time rejection belongs in tests/negative instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPILER="${1:-$PROJECT_ROOT/bin/nanoc_c}"
WORK="$(mktemp -d "${TMPDIR:-/tmp}/nanolang-runtime-failures.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT

run_failure() {
    local source="$1"
    local expected="$2"
    local binary="$WORK/$(basename "$source" .nano)"
    local log="$binary.log"

    perl -e 'alarm 60; exec @ARGV' "$COMPILER" "$source" -o "$binary" \
        >"$WORK/compile.log" 2>&1
    test -x "$binary"

    set +e
    perl -e 'alarm 10; exec @ARGV' "$binary" >"$log" 2>&1
    local status=$?
    set -e

    if [ "$status" -eq 0 ]; then
        echo "FAIL: $(basename "$source") succeeded; expected runtime failure" >&2
        return 1
    fi
    if ! grep -Eiq "$expected" "$log"; then
        echo "FAIL: $(basename "$source") did not report '$expected'" >&2
        cat "$log" >&2
        return 1
    fi
    echo "  ✓ $(basename "$source") failed at the documented boundary"
}

run_failure "$SCRIPT_DIR/requires_fail_test.nano" 'Contract violation at line [0-9]+: \(> x 0\)'
