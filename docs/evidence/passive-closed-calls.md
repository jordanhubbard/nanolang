# Closed scalar calls in passive records

I extend version-2 node validation with concrete local calls. I derive scalar
arguments from guarded inputs, constants, scalar producers and prior verified
node results. I inspect reachable callee code, intersect definite local
initialization across branches and loops, and check exact call/return stack
counts. I memoize completed summaries and refuse recursive call graphs. I do
not use a signature annotation as a proof of the actual value.

The bounded implementation lives in `src/nanoisa/passive_calls.inc`; its limits
and admitted operations are documented in `docs/NANOISA_PASSIVE.md`. Local
reassignment is permitted. Foreign and indirect calls, captures, aggregate
operations, global state and observable printing are refused. Metadata-free
modules and version-1 records retain their existing behavior.

I ran `make test-passive-metadata test-verifier test-disasm-roundtrip`:

- 15 Python methods passed, including six new closed-call methods.
- 249 retained C passive checks, 96 verifier checks and 210 canonical roundtrip
  checks passed.
- Nested square/cube calls, a counted local loop, a branch assigning its result
  on both paths, zero-argument and void-result calls execute identically in VM
  and strict standalone native products.
- The two-argument arctangent loop verifies, roundtrips exactly and prints
  `3.14159` in VM. Native translation still refuses `CAST_FLOAT`. I retain this
  explicit boundary as `task_3d6c3314d8924dd2b0aca679647e7048`.

I also compiled `passive.c` (including the new checker) with ASan and UBSan at
`-O0`, linked it into a separate assembler using the other ordinary objects,
and reran all six new methods without sanitizer findings. This instruments the
new checker; it is not a claim that every linked object was instrumented.

Evidence logs are `/tmp/nanolang-passive-closed-final3.log` and
`/tmp/nanolang-passive-closed-sanitizer-final.log`. The retained exact arctangent
module is `/tmp/nanolang-passive-arctan.nvm`.

This completes only verifier prerequisite
`task_7689d21c7fac465287e90a425ea4b9eb`. Source frontend task
`task_e0c6fd18cfbb49c78e4c509d8444417e` remains open: its cutover must preserve
existing callable `par` fixtures, including paired native arctangent execution.
The isolated scalar frontend draft has not been merged. Foreign intrinsic
identity, broader external-input proofs and `flow` extraction remain separate.
