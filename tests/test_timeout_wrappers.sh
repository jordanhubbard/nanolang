#!/usr/bin/env bash
set -eu

# I test the Make-expanded recipes, not copies of their implementation.
exec python3 "$(dirname "${BASH_SOURCE[0]}")/test_make_timeouts.py" "$@"
