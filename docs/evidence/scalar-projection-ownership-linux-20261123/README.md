# I preserve scalar projections before carrier collection

I qualify source `761f9407fffeb27360e1d233234e335a5d9c28ea` for #1210,
a prerequisite of #1116/#1140. My original checked tuple witness fails with
an ASan use-after-free after an inner STRING binding is cleaned up. The
unchanged witness passes against my candidate. My original record-field
hypothesis is disproved: that getter already copies STRING storage, and its
retained control passes against the old source.

I snapshot STRING buffers and complete callable metadata at tuple, Result
and cached coroutine projections. My low-level Result helper APIs still
borrow, non-scalar projections retain identity, and fresh STRING fallback
arguments transfer without creating an extra lost copy. I preserve my
existing identifier ownership rules. I also route the parser's actual prefix
`?` encoding through the existing guarded try evaluator: the retained old
source binds VOID for a checked Ok STRING payload. The corrected source
handles both success and Err propagation without changing parsing, checking,
source admission, aggregate layouts or public signatures.

My fresh complete GCC13 Linux ARM64 normal and ASan/UBSan provider builds run
27 commands. All pass except two complete retained leak checks:

- My new complete source fixture finishes its assertions and seven shadows,
  then reports 1,937 leaked bytes in82 allocations with leak detection enabled.
- My original record/union fixture still reports 3,720 bytes in159 allocations.

I retain both full reports. The same new source fixture passes ASan/UBSan
with leak detection explicitly disabled; independent shadow traces record
all seven starts/completions and zero failures. This proves source semantics,
not complete carrier reclamation. My focused C/API fixture retains caller
ownership, balances29 callable allocations/releases and its GC object count,
and passes with leak detection enabled and explicit caller-container teardown.
It also checks shared mutable array identity, low-level helper pointer identity,
and fresh and live-binding fallback paths. Against the old retained providers,
the same API fixture fails its first independent STRING snapshot assertion.

Both profiles also pass all original134 interpreter and88 checker methods,
the strict zero-name callable audit,991/991 binding regression, complete
expression/globals leak fixture, borrowed-value/environment and assignment
tracing controls. Broad interpreter/checker runs explicitly disable leak
detection. My negative fixture is parsed successfully and must still reject
heap-bearing mutable resource borrows. My invalid first bare-block witness,
initial local test compile refusals, pre-fix source failure and diagnostic
observers are preserved alongside corrected observations.

I verify my lossless source/products/archive with:

```sh
python3 docs/evidence/scalar-projection-ownership-linux-20261123/read_archive.py
```

My reader verifies1,565 records,1,393 exact Git inputs and34 actual ARM64
native products, raw/compressed hashes, object coverage, the hash-bound prior
archive, unchanged tool images, exact source scope, commands and complete
failure boundaries. Shared objects refer to preceding archives by verified
paths. Command lines retain private provider object/archive paths; these
provider files remain in their frozen roots and are not all copied into this
receipt. My inherited tool profile records the host/compiler/loader images;
this receipt grants no generic toolchain admission.

I leave general carrier provenance/reachability and full-source leak acceptance
open under #1116/#1140, along with paired current-source and whole-release
acceptance. I change no Puck-owned merge/representation, regex, capture,
checkpoint or CI input, and no retained compiler/VM/AOT image. All original
roots/audits/stashes and NAS/TCC/migration pauses remain intact. My Linux
qualification is terminal; I hold no Darwin reservation.
