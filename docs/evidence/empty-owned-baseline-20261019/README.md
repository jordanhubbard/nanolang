# I retain the unchanged empty-resource baseline

I capture exactly PREFIX plus `OwnedRecordPatterns.test_empty_resource_pattern`,
as the existing paired source acceptance test does. The current seed and both
installed stages produce ordinary main/close with2/3 locals; a separately pinned
retained C-hosted original-shadow producer emits entry/close/main with1/3/2
locals. All four original modules verify and execute with exit0/empty stdout.
No new shadow driver or compiler is built.

Handle is layout0; Empty is the distinct zero-field layout1. Ordinary main
executes OWN_PACK1, STORE0, MOVE0, STORE1, PUSH_I640, OWN_UNPACK1, RET. There are
no field POPs for the empty unpack. The original shadow also executes the
prefixed close shadow and calls main. Its three-function cardinality is kept
separate from ordinary two-function output.

All four facts queries refuse `I require a verified bounded resource module
with exact fields.` Both reconstruction targets retain their prior-output
sentinels for every module. This is actual original runtime acceptance plus
actual facts/reconstruction refusal, not merely a source-level prediction.
I retain exact dumps, source, every command/status/output and module-summary.
All ten selected source/tool/driver hashes match before/after. Binaries remain
at `/private/tmp/empty-owned-baseline-20261019`. Worker health before this
bounded run was218GiB free disk,69% free memory and44% CPU idle.

No production/admission change is included. The next interface must preserve
zero-field nominal layout identity and once-only owner lifetime in C and Nano,
including explicit portable C representation. Additional translator guards or
output limitations remain unmeasured until reviewed implementation. Full #999
and exact candidate platform qualification remain open.
