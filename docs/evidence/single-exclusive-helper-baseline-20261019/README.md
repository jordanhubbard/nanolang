# I retain the single-exclusive helper-local baseline

I run the exact `former_refusal` substitution recorded in
`NANOISA_SINGLE_EXCLUSIVE_HELPER_LOCAL_DESIGN.md` after helper integration
`4bee9c151`, with aggregate providers frozen. My three ordinary producers and
one separately pinned existing C-hosted original-shadow driver all produce
verified modules that execute with exit0 and empty stdout. Ordinary main/bump
and independent shadow-entry/bump each have exactly two functions and52/7 locals.
All four facts queries succeed. Both reconstruction languages refuse all four
inputs with `I require bounded exclusive helper control`, preserving the exact
previous-output sentinel. This is an analyzer refusal, not a facts refusal.

My actual helper dump constructs Leaf0, stores slot3, moves into slot4, then
uses two complete ephemeral shared regions at descriptor1 on slot4 (fields0
and1), storing snapshots into5/6. It then reads inherited descriptor0 field0,
adds1, writes descriptor0 field0, reads it again, unpacks owner4, discards both
scalar fields and returns. Source destructuring therefore requires local
descriptor1 observations despite no explicit field-read syntax. The formal
must remain valid across both regions and final local consumption.

`module-summary.json` retains exact function headers/helper instructions;
ordinary and original-shadow dumps, facts, every command/status/stdout/stderr,
source and bounded runner are retained. `pins-before.json` and `pins-after.json`
match for all ten selected source/tool/driver files. The original-shadow driver
is C-hosted retained code, not three newly compiled shadow producers. Raw
modules remain in `/private/tmp/single-exclusive-helper-baseline-20261019`.
The worker had221GiB free disk,66% free memory and43% CPU idle before this
short serial run. No compiler, runtime, verifier or reconstruction code changed.

I have not measured reconstructed source counts, run reconstructed native
products or admitted this profile. Existing implementation review must cover
single-formal read/write tuples, descriptor1 observations, separate local
origins and total work before reusing the single-exclusive consistency witness.
Fresh/platform qualification and full #999 remain open.
