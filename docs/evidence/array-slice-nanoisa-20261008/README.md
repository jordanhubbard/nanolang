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
