#!/usr/bin/env bash
# I run the bounded experiment without installing a host Sail toolchain.
set -euo pipefail
mode=${1:---execute}
case "$mode" in
    --execute|--rocq-export-only|--rocq-check) ;;
    *) echo 'Usage: check_sail_container.sh [--execute|--rocq-export-only|--rocq-check]' >&2; exit 2 ;;
esac
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
sail_tmp=$(mktemp -d "${TMPDIR:-/tmp}/nanolang-sail.XXXXXX")
trap 'rm -rf -- "$sail_tmp"' EXIT
if [[ "$mode" == --execute ]]; then
    python3 "$repo_root/scripts/sail_decode_cases.py" "$sail_tmp"
    python3 "$repo_root/scripts/sail_vm_cases.py" "$sail_tmp"
fi
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
    --env SAIL_TRIAL_MODE="$mode" \
    --env PATH=/opt/sail/bin:/usr/local/bin:/usr/bin:/bin \
    --mount "type=bind,source=$sail_tmp/sail,target=/opt/sail,readonly" \
    --mount "type=bind,source=$repo_root/formal/sail,target=/source,readonly" \
    --mount "type=bind,source=$repo_root/formal/check_assumptions.sh,target=/check_assumptions.sh,readonly" \
    --mount "type=bind,source=$sail_tmp,target=/cases,readonly" \
    rocq/rocq-prover@sha256:3ed9c46fa02e9fd7748a959abebb11b36f8818d10738dcc93edc67354c313ce2 \
    bash -c '
        set -euo pipefail
        cd /tmp
        sail --version
        if [ "$SAIL_TRIAL_MODE" != --execute ]; then
            sail --no-memo-z3 --rocq --rocq-lib-style stdpp /source/stack_slice.sail -o stack_slice
            test -s stack_slice_types.v
            test -s stack_slice.v
            if [ "$SAIL_TRIAL_MODE" = --rocq-export-only ]; then
                cat stack_slice_types.v stack_slice.v
                echo "I generated Rocq definitions; I have not checked them with Rocq."
                exit 0
            fi
            opam install -y coq.9.0.1 coq-core.9.0.1 coq-stdlib.9.0.0 \
                coq-sail-stdpp.0.20.2 coq-stdpp-bitvector.1.12.0 coq-stdpp.1.12.0
            opam exec -- coqc stack_slice_types.v
            opam exec -- coqc stack_slice.v
            cp /source/StackSliceProofs.v .
            opam exec -- coqc StackSliceProofs.v > assumptions.log
            cat assumptions.log
            bash /check_assumptions.sh StackSliceProofs.v assumptions.log \
                nop_identity push_then_pop dup_then_pop swap_involution \
                dup_underflow pop_underflow swap_empty_underflow swap_singleton_underflow \
                execute_frame_extension
            opam exec -- coqchk -silent StackSliceProofs
            echo "I checked bounded stack-model lemmas, not VM refinement."
            exit 0
        fi
        sail --no-memo-z3 --just-check /source/stack_slice.sail /source/smoke.sail
        sail --no-memo-z3 -c /source/stack_slice.sail /source/smoke.sail -o stack_slice
        cc stack_slice.c /opt/sail/share/sail/lib/*.c \
            -I/opt/sail/share/sail/lib -lgmp -l:libz.so.1 -o stack_slice
        ./stack_slice
        sail --no-memo-z3 -c /source/stack_slice.sail /cases/decode_cases.sail -o decode_cases
        cc decode_cases.c /opt/sail/share/sail/lib/*.c \
            -I/opt/sail/share/sail/lib -lgmp -l:libz.so.1 -o decode_cases
        ./decode_cases
        sail --no-memo-z3 -c /source/stack_slice.sail /cases/vm_cases.sail -o vm_cases
        cc vm_cases.c /opt/sail/share/sail/lib/*.c \
            -I/opt/sail/share/sail/lib -lgmp -l:libz.so.1 -o vm_cases
        ./vm_cases
    '
