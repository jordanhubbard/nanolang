#!/usr/bin/env bash
# nanoc --locale / --print-locale. Does not compile. Does not claim catalogs.
set -euo pipefail

NANOC="${1:?usage: test_locale_cli.sh <nanoc>}"
export NANO_LOCALE=
unset NANO_LOCALE || true

expect_key() {
    local text="$1" key="$2"
    printf '%s' "$text" | grep -q "^${key}: " || {
        echo "missing ${key} in:" >&2
        printf '%s\n' "$text" >&2
        exit 1
    }
}

out="$("$NANOC" --locale en --print-locale)"
expect_key "$out" "tag"
expect_key "$out" "language"
expect_key "$out" "script"
expect_key "$out" "region"
expect_key "$out" "variant"
expect_key "$out" "direction"
expect_key "$out" "encoding"
expect_key "$out" "collation"
expect_key "$out" "source"
expect_key "$out" "fallback"
printf '%s' "$out" | grep -q '^tag: en$' || { echo "expected tag: en"; exit 1; }
printf '%s' "$out" | grep -q '^source: cli$' || { echo "expected source: cli"; exit 1; }
printf '%s' "$out" | grep -q '^encoding: utf-8$' || { echo "expected utf-8"; exit 1; }

out="$("$NANOC" --locale ar --print-locale)"
printf '%s' "$out" | grep -q '^tag: ar$' || { echo "expected tag: ar"; exit 1; }
printf '%s' "$out" | grep -q '^direction: rtl$' || { echo "expected rtl"; exit 1; }
printf '%s' "$out" | grep -q '^fallback: ar en$' || { echo "expected fallback ar en"; exit 1; }

if "$NANOC" --locale 'en--US' --print-locale >/dev/null 2>&1; then
    echo "invalid --locale should fail closed" >&2
    exit 1
fi

if "$NANOC" --locale >/dev/null 2>&1; then
    echo "missing --locale argument should fail" >&2
    exit 1
fi

echo "nanoc locale CLI: PASS"
