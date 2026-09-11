#!/usr/bin/env bash
# I run the bounded Sail experiment without installing a host toolchain.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sail_tmp=$(mktemp -d "${TMPDIR:-/tmp}/nanolang-sail.XXXXXX")
trap 'rm -rf -- "$sail_tmp"' EXIT
archive="$sail_tmp/sail.tar.gz"
curl --fail --location --silent --show-error --retry 2 \
    https://github.com/rems-project/sail/releases/download/0.20.2-binary/sail-Linux-x86_64.tar.gz \
    -o "$archive"
expected=26b59bcab2d66e9f220d317dfe45f8b09170ed70e59a824553d6f525134d1ff6
actual=$(shasum -a 256 "$archive" | awk '{print $1}')
if [[ "$actual" != "$expected" ]]; then
    echo 'I reject the Sail archive: its checksum does not match.' >&2
    exit 1
fi
tar -xzf "$archive" -C "$sail_tmp"
docker run --rm --platform linux/amd64 \
    --env PATH=/opt/sail/bin:/usr/local/bin:/usr/bin:/bin \
    --mount "type=bind,source=$sail_tmp/sail,target=/opt/sail,readonly" \
    --mount "type=bind,source=$repo_root/formal/sail,target=/source,readonly" \
    rocq/rocq-prover@sha256:3ed9c46fa02e9fd7748a959abebb11b36f8818d10738dcc93edc67354c313ce2 \
    bash -c '
        set -euo pipefail
        cd /tmp
        sail --version
        sail --no-memo-z3 --just-check /source/stack_slice.sail
        sail --no-memo-z3 -c /source/stack_slice.sail -o stack_slice
        cc stack_slice.c /opt/sail/share/sail/lib/*.c \
            -I/opt/sail/share/sail/lib -lgmp -l:libz.so.1 -o stack_slice
        ./stack_slice
    '
