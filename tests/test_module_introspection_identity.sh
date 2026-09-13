#!/usr/bin/env bash
# I require an explicit compiler so this test cannot silently change stages.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"
compiler=${1:?I require a compiler path.}
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/nanolang-module-identity.XXXXXX")
trap 'rm -rf -- "$test_dir"' EXIT
if "$compiler" tests/negative/neg_module_introspection_identity.nano -o "$test_dir/program" >"$test_dir/log" 2>&1; then
    echo 'I accepted two modules with the same introspection identity.' >&2
    exit 1
fi
if ! grep -Fq 'I reject ambiguous module introspection identity: duplicate_identity' "$test_dir/log"; then
    sed -n '1,40p' "$test_dir/log" >&2
    exit 1
fi
echo 'I rejected ambiguous module introspection identity.'
