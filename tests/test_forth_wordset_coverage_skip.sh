#!/bin/bash
# Coverage instrumentation must skip Jackson word-set REFILL (TEST_TIMEOUT).
# Pin and INCLUDE-gap still run. Does not claim Core.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"

covered=$(make -n --no-print-directory test-impl \
    CFLAGS='-Wall -fprofile-arcs -ftest-coverage' 2>/dev/null)

if ! grep -q 'Skipping Jackson word-set REFILL under coverage instrumentation.' <<<"$covered"; then
    echo "FAIL: coverage dry-run did not skip Jackson word-set REFILL"
    exit 1
fi
if ! grep -q 'test-forth-jackson' <<<"$covered"; then
    echo "FAIL: coverage dry-run omitted test-forth-jackson"
    exit 1
fi
if grep -q 'test-forth-wordsets' <<<"$covered"; then
    echo "FAIL: coverage dry-run still invoked test-forth-wordsets"
    exit 1
fi

plain=$(make -n --no-print-directory test-impl 2>/dev/null)
if ! grep -q 'test-forth-wordsets' <<<"$plain"; then
    echo "FAIL: default dry-run omitted test-forth-wordsets"
    exit 1
fi
if ! grep -q 'test-forth-jackson' <<<"$plain"; then
    echo "FAIL: default dry-run omitted test-forth-jackson"
    exit 1
fi

echo "PASS Forth word-set skip under coverage flags"
