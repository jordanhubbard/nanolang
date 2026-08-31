#!/usr/bin/env bash
set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
binary=/tmp/nano_dynamic_profile
profile=/tmp/nano_dynamic_profile.nano.prof
stderr_on=/tmp/nano_dynamic_profile.on.stderr
stderr_off=/tmp/nano_dynamic_profile.off.stderr
trap 'rm -f "$binary" "$profile" "$stderr_on" "$stderr_off"' EXIT

"$root/bin/nanoc" "$root/tests/unit/test_profile_runtime.nano" \
    --profile -o "$binary" >/tmp/nano_dynamic_profile.compile 2>&1

NANO_PROFILE=1 "$binary" >/dev/null 2>"$stderr_on"
grep -q -- 'NanoLang Profile Report' "$stderr_on"

NANO_PROFILE=0 "$binary" >/dev/null 2>"$stderr_off"
if grep -q -- 'NanoLang Profile Report' "$stderr_off"; then
    printf 'profile report was not disabled at runtime\n' >&2
    exit 1
fi

printf 'dynamic generated-C profiling checks passed\n'
