#!/bin/sh
set -eu

# I reuse private fixtures and require identity diagnostics, not arbitrary failure.
if [ -n "${NANOC_SELFHOST:-}" ]; then
    NANOLANG_SELFHOST_COMPILER="$(cd "$(dirname "$NANOC_SELFHOST")" && pwd)/$(basename "$NANOC_SELFHOST")"
    export NANOLANG_SELFHOST_COMPILER
fi
cd "$(dirname "$0")/.."
exec python3 -m unittest \
    tests.test_selfhost_cli.SelfhostCliTests.test_source_aliases_are_rejected_before_writes \
    tests.test_selfhost_cli.SelfhostCliTests.test_diagnostic_alias_is_rejected_before_parse_error \
    tests.test_selfhost_cli.SelfhostCliTests.test_uncheckable_destination_identity_fails_closed
