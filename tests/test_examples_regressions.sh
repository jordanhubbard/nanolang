#!/bin/bash

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

COMPILER="bin/nanoc_c"
TIMEOUT_SECONDS="${EXAMPLE_REGRESSION_TIMEOUT:-20}"
work_dir="$(mktemp -d "${TMPDIR:-/tmp}/example_regressions.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

if [ ! -x "$COMPILER" ]; then
    echo "ERROR: $COMPILER is not built"
    exit 1
fi

export NANO_MODULE_PATH="${NANO_MODULE_PATH:-$REPO_ROOT/modules}"

failures=0
compile_example() {
    local source="$1"
    local name="${source##*/}"
    name="${name%.nano}"
    local log="$work_dir/$name.log"

    echo "Compiling $source..."
    if perl -e "alarm $TIMEOUT_SECONDS; exec @ARGV" \
            "$COMPILER" "$source" -o "$work_dir/$name" >"$log" 2>&1; then
        echo "  ✓ $source"
        return
    else
        local status=$?
    fi

    echo "  ✗ $source (exit $status)"
    if [ "$status" -eq 142 ]; then
        echo "    compilation exceeded ${TIMEOUT_SECONDS}s"
    else
        while IFS= read -r line; do
            echo "    $line"
        done < "$log"
    fi
    failures=$((failures + 1))
}

compile_example examples/graphics/sdl_game_of_life.nano
compile_example examples/audio/sdl_nanoamp.nano

if [ "$failures" -ne 0 ]; then
    echo "ERROR: $failures example regression(s) failed"
    exit 1
fi

echo "Example regressions passed"
