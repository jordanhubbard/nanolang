#!/usr/bin/env bash
# I build a fresh copy and independently recheck its compiled proof libraries.
set -euo pipefail
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
proof_image=rocq/rocq-prover@sha256:3ed9c46fa02e9fd7748a959abebb11b36f8818d10738dcc93edc67354c313ce2
docker run --rm --platform linux/amd64 \
    --mount "type=bind,source=$repo_root/formal,target=/source,readonly" \
    "$proof_image" bash -lc '
        set -euo pipefail
        cp -R /source /tmp/nanocore
        cd /tmp/nanocore
        eval "$(opam env)"
        rocq --version
        make -B check COQC="rocq compile" COQCHK="rocq chk"
    '
