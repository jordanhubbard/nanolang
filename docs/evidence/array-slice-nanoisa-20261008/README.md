# My array-slice retirement baseline

I retain the baseline for [#979](https://github.com/jordanhubbard/nanolang/issues/979).
My tools are the previously qualified lexical-correction binaries retained at
`/private/tmp/nanolang-retirement-20261008/bin`; `manifest.json` records their hashes.
The worktree has since changed branches, so its current HEAD is not a build receipt.

My minimal integer-array slice emits and executes through NanoVirt. My retained
Stage2 refuses it with `undefined function array_slice`. This is a product gap,
not a passing retirement test. `minimal.json` records the observed exits.

My expanded probe covers signed extremes, exact negative-zero/NaN/subnormal bits,
u8, and records retaining child arrays. After correcting an invalid `0..4` loop
in the probe to `(range 0 4)`, NanoVirt emits and executes it. Stage2 rejects
`array<u8> = [255]` in checking before emission. I preserve both initial syntax
errors and corrected logs; the initial syntax errors are my harness mistake.
The emitted native path and fresh compiler parity are not yet qualified.

I retain the old helper-extraction test until backend-appropriate coverage
replaces every boundary, including null input behavior.

## My first lowering candidate

My C-seed-built emitter driver completes its build and shadows (exit zero).
My new four-method component suite runs seven positive source cases through
NanoVirt and the actual self-hosted emitter, then verifies and executes VM
output before requesting sanitized native translation. It also checks four
invalid-operand refusals. The component terminal has fourteen failed subcases:
native translation rejects ARR_SLICE; self-hosted byte-array and nested-array
cases refuse before translation. The invalid-operand method passes.

I retain candidate source hashes and the complete terminal. This is an
implementation checkpoint with demonstrated remaining gaps, not qualification.
The original helper-extraction test remains intact.
