#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

make_bin=${MAKE_BIN:-make}
object=obj/nanovm/value.o
dependency_file=obj/nanovm/value.d
indirect_header=src/nanovm/../nanoisa/isa.h
trigger=$(mktemp "${TMPDIR:-/tmp}/nanolang-header-dependency.XXXXXX")
trap 'rm -f "$trigger"' EXIT

rm -f "$object" "$dependency_file"
"$make_bin" --no-print-directory "$object" >/dev/null

if [[ ! -s "$dependency_file" ]]; then
    echo "I did not generate $dependency_file" >&2
    exit 1
fi

if ! grep -Fq "$indirect_header" "$dependency_file"; then
    echo "I did not record the indirect header $indirect_header" >&2
    exit 1
fi

# Add a newer synthetic header to the generated dependency file. This proves
# that Makefile.gnu reads .d files, without changing a source header's mtime or
# racing another test that may be reading it.
printf '%s: %s\n' "$object" "$trigger" >>"$dependency_file"
# The system make on older macOS compares mtimes at one-second resolution.
sleep 1
touch "$trigger"
if "$make_bin" --no-print-directory -q "$object"; then
    echo "I trusted $object after one of its recorded headers changed" >&2
    exit 1
else
    status=$?
    if [[ $status -ne 1 ]]; then
        echo "I could not query $object after its dependency changed (status $status)" >&2
        exit "$status"
    fi
fi

"$make_bin" --no-print-directory "$object" >/dev/null
if ! "$make_bin" --no-print-directory -q "$object"; then
    echo "I still consider $object stale after rebuilding it" >&2
    exit 1
fi

echo "I rebuild C objects when their recorded headers change."
