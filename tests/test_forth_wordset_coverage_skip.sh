#!/bin/bash
# Coverage instrumentation must skip Jackson word-set REFILL (TEST_TIMEOUT)
# and Forth PTY/IDE smoke (gcov stalls the REPL banner). Pin and INCLUDE-gap
# still run. Does not claim Core.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

# Isolate from a parent `make test CFLAGS=...-fprofile-arcs...` so the
# default dry-run does not inherit coverage flags via MAKEFLAGS.
covered=$(MAKEFLAGS= make -n --no-print-directory test-impl \
    CFLAGS='-Wall -fprofile-arcs -ftest-coverage' 2>/dev/null)

if ! grep -q 'Skipping Jackson word-set REFILL, Forth PTY, and IDE smoke under coverage instrumentation.' <<<"$covered"; then
    echo "FAIL: coverage dry-run did not skip Jackson word-set REFILL / PTY / IDE"
    exit 1
fi
if ! grep -q 'test-forth-jackson' <<<"$covered"; then
    echo "FAIL: coverage dry-run omitted test-forth-jackson"
    exit 1
fi
for t in test-forth-wordsets test-forth-pty test-forth-ide-smoke; do
    if grep -q "$t" <<<"$covered"; then
        echo "FAIL: coverage dry-run still invoked $t"
        exit 1
    fi
done

plain=$(MAKEFLAGS= make -n --no-print-directory test-impl \
    CFLAGS='-Wall -Wextra -Werror -std=c99 -g -Isrc -D_GNU_SOURCE' 2>/dev/null)
for t in test-forth-wordsets test-forth-jackson test-forth-pty test-forth-ide-smoke; do
    if ! grep -q "$t" <<<"$plain"; then
        echo "FAIL: default dry-run omitted $t"
        exit 1
    fi
done

echo "PASS Forth word-set skip under coverage flags"
