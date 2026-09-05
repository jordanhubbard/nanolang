#!/usr/bin/env bash
# Run language examples under bin/nano. The six KNOWN_LIMITATIONS files must
# exit 0 (or print SKIP:). Other files may skip when interactive.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

NANO="${NANO:-./bin/nano}"
TIMEOUT_SECONDS="${INTERPRETER_EXAMPLE_TIMEOUT:-20}"

if [ ! -x "$NANO" ]; then
    echo "ERROR: $NANO is not built (run make build)"
    exit 1
fi

export NANO_MODULE_PATH="${NANO_MODULE_PATH:-$REPO_ROOT/modules}"

must_pass=(
    examples/language/nl_random_sentence.nano
    examples/language/nl_primes_sieve.nano
    examples/language/nl_primes.nano
    examples/language/nl_game_of_life.nano
    examples/language/nl_dispatch_counter.nano
    examples/language/nl_dispatch_pipeline.nano
    examples/language/nl_dispatch_stats.nano
)

failures=0

run_one() {
    local source="$1"
    local require_pass="$2"
    local log
    log="$(mktemp "${TMPDIR:-/tmp}/nano_ex.XXXXXX")"
    local status=0

    perl -e "alarm $TIMEOUT_SECONDS; exec @ARGV" \
        "$NANO" "$source" </dev/null >"$log" 2>&1 || status=$?

    if grep -q '^SKIP:' "$log"; then
        echo "  skip $source"
        rm -f "$log"
        return 0
    fi

    if [ "$status" -eq 142 ]; then
        if [ "$require_pass" = "1" ]; then
            echo "  FAIL $source (timeout ${TIMEOUT_SECONDS}s)"
            failures=$((failures + 1))
        else
            echo "  skip $source (interpreter timed out; likely interactive)"
        fi
        rm -f "$log"
        return 0
    fi

    if [ "$status" -ne 0 ]; then
        echo "  FAIL $source (exit $status)"
        sed 's/^/    /' "$log" | sed -n '1,20p'
        failures=$((failures + 1))
        rm -f "$log"
        return 0
    fi

    echo "  pass $source"
    rm -f "$log"
}

echo "Interpreter language examples (must pass)..."
for src in "${must_pass[@]}"; do
    run_one "$src" 1
done

if [ "$failures" -ne 0 ]; then
    echo "ERROR: $failures required interpreter example(s) failed"
    exit 1
fi

echo "Required interpreter examples passed"
exit 0
