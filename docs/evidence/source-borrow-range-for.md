# My owned-source range-loop evidence

MAC `task_212a2ec1f7374734b90a23abf00a3211`, PR688.

I recorded the range contract before edits at `8a1e578e`. My first paired
implementation passed fresh bootstrap and 25 of 26 source methods, but its
one-bound fixture was rejected with E003 by the C frontend. I had inferred
source admission from a broader raw-lowering path. The documented source
builtin and interpreter loop require `range(start, end)`; I corrected both
producers to exactly two bounds, kept explicit zero starts in positive cases,
and added one-bound refusal. I retain the first log at
`/tmp/nanolang-borrow-range-for-gate.log` and the separate cross-layer audit
`task_a1c30f5fe3a740c9906e150200d05af8`. I did not expand the checker or weaken
endpoint/ownership assertions.

My corrected integrated checkpoint `26e11f38`, based on main through PR689,
passed fresh bootstrap and all 26 methods of `make -j4
test-source-borrow-emission` in 288.560 seconds. C, selfhost-produced emitters,
and canonical Stage1/Stage2 outputs retain exact instructions, layouts,
ownership and lexical-name metadata. Name stripping and all selected shadows
retain VM/sanitized native execution. The new fixture proves ordered once-only
bound-call effects, lexical restoration, nested loop targets, continue/break,
zero/reversed ranges, exactly one iteration below INT64_MAX, and return paths.
Six source refusal controls retain prior output.

My integrated `make -j4 test-affine-bytecode test-owned-assertions` passes
441 and 751 affine bytecode checks plus 959 assertion lifecycle checks.
I retain `/tmp/nanolang-borrow-range-for-integrated.log` and
`/tmp/nanolang-borrow-range-for-integrated-authority.log`. Production remained
unchanged during this corrected qualification. A subsequent main restack
contains reconstruction-script/docs changes only; my compiler and test sources
remain identical to this measured checkpoint.

My live ledger audit records `20048` and `c60a` as completed bounded historical
work. Full normative One-IR `ed702`, borrow `718` and release-equivalence `28f2`
remain open. This slice adds neither runtime/verifier authority nor general
array iteration, helper-owned locals, deeper calls or implicit resource drops.
No historical failed compiler artifact was replayed.
