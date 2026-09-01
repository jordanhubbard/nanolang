#!/usr/bin/env bash
# Generated-C tracing is dynamically selectable at process startup via NANO_TRACE
# and shares one hook mechanism with --profile. A disabled trace does no work.
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
binary=/tmp/nano_dynamic_trace
trace_on=/tmp/nano_dynamic_trace.on.stderr
trace_off=/tmp/nano_dynamic_trace.off.stderr
trap 'rm -f "$binary" "$trace_on" "$trace_off"' EXIT

"$root/bin/nanoc" "$root/tests/unit/test_profile_runtime.nano" \
    --trace -o "$binary" >/tmp/nano_dynamic_trace.compile 2>&1

# Enabled by default in a --trace binary.
"$binary" >/dev/null 2>"$trace_on"
grep -q -- '-> nl_main' "$trace_on"
grep -q -- '<- nl_main' "$trace_on"

# NANO_TRACE=0 disables collection at runtime without recompiling.
NANO_TRACE=0 "$binary" >/dev/null 2>"$trace_off"
if grep -q -- '-> nl_' "$trace_off"; then
    printf 'trace output was not disabled at runtime\n' >&2
    exit 1
fi

# Explicit NANO_TRACE=1 keeps tracing on.
NANO_TRACE=1 "$binary" >/dev/null 2>"$trace_on"
grep -q -- '-> nl_count_down' "$trace_on"

printf 'dynamic generated-C tracing checks passed\n'
