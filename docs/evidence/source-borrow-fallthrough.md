# My borrowed-source scalar return acceptance

I implement stable-owner scalar return paths under
`task_b077ad608868462da669b5a7a427567d`, with explicit destructive-pattern
holder provenance under `task_6a55c8c1e40a4923804e20b450d32cdf`.
My [contract](../NANOISA_SOURCE_BORROW_FALLTHROUGH.md) does not admit branch-local
resource movement or implicit source drops. That continuation remains
`task_d74d8a4fb4a048a786666195eaa4e8d5`.

My source began on main `79e156ac`. The first path-specific checkpoint was
`965f9d2d`; the disposal-provenance prerequisite is `2decca38`. I retain the
original positive refusal: 18 of 19 source methods passed, but the new entry
return was refused after explicit leaf destructuring. The existing lowering
moves a source owner into a hidden leaf-pattern holder and defers its unpack
until terminal cleanup. The initial early-return guard could not distinguish
that holder from a live source owner. The retained log is
`/tmp/nanolang-borrow-fallthrough-gates.log`; this is a demonstrated new-guard
limitation, not an infrastructure classification.

I now mark pending disposal only after complete, distinct, exact pattern
validation. Every new slot starts unmarked. Cleanup/unpack clears the flag;
branch snapshots restore it with liveness. I preserve the moved/dead source
owner and do not infer authority from a generated name. Borrowed formals
remain caller-owned. The verifier, runtime, wire contract and existing leaf
projection instructions are unchanged.

My paired source gate checks entry then/else returns, a missing else with
continuation, entered and zero-iteration loops, returning helper mutations,
Boolean helper results and complete explicit disposal. Exact code, ownership,
layouts and advisory names are compared for C-seed and selfhost producers,
including Stage1/Stage2-built tools and canonical publication. Every selected
shadow remains checked. Ordinary refusals cover nested shadow returns, live
owners, missing returns, owner moves and incomplete/duplicate patterns.
VM and ASan/UBSan native execution, including stripped advisory names, remain
required. I do not replay any held product compiler artifact.

Logs:

- `/tmp/nanolang-borrow-fallthrough-bootstrap.log` (first checkpoint)
- `/tmp/nanolang-borrow-fallthrough-gates.log` (retained refusal)
- `/tmp/nanolang-borrow-fallthrough-disposal-bootstrap.log`

My corrected fresh bootstrap passed. The final source gate passed all 19
methods in 182.416 seconds, including seven positive return variants and all
six dedicated refusal cases. The same invocation passed 441 ordinary and 751
allocation-instrumented affine bytecode checks, 959 owned assertion lifecycle
checks and 123 lexical local-name codec checks. Native fixtures used
ASan/UBSan with leak detection. Final log:
`/tmp/nanolang-borrow-fallthrough-disposal-gates.log`.

I ran `make -j8 -o bootstrap test-source-borrow-emission test-affine-bytecode test-owned-assertions`
after the fresh corrected bootstrap; no test method was skipped. The source
pin stayed on main `79e156ac` plus this slice during acceptance. Later main
merges were not copied into the running tree.
