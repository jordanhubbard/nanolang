# My explicit numeric-union shape acceptance

I recorded [my contract](../NATIVE_NUMERIC_UNION_SHAPES.md) before changing
code under `task_87a7b44d10e240e999485d12aeecaca2`. Production `5713149c`
restacks as `6300aee3` onto the merged native invariant diagnostics. I preserve
those diagnostics and do not change runtime arithmetic or its tag checks.

NUMERIC is a distinct INT|FLOAT shape leaf. An explicitly numeric OPTIONAL
payload accepts directed injections from either exact member. Exact unification
still refuses INT/FLOAT/NUMERIC mismatches, and NUMERIC has no child edges.
I neither infer a union from conflicting exact types nor alter the meaning of
existing OPTIONAL(INT), OPTIONAL(STRING), array or map payload constraints.

Numeric arithmetic outputs carry closed provenance. Stable branch plans,
local writes and actual call arguments combine only that provenance and exact
numeric members. Existing optional absence retains its void tag. Unproved,
U8, Boolean or heap provenance cannot enter a proved numeric join/local/parameter.
Parameter provenance starts empty and is populated by actual callers; an
unresolved load still receives the existing unknown marker. This avoids treating
an absent argument fact as a proof or permanently contaminating a later resolved
caller with a provisional unknown kind.

My first focused run exposed that the older boxed/string join fallback admitted
U8 beside a numeric result. I retained its failure in
`/tmp/nanolang-numeric-union-focused.log` and added explicit provenance checks at
joins, local writes and call-parameter merging. The corrected negative controls
reject U8, bool, string and unknown tagged input and preserve previous output.

## My measured gates

- 1,139 solver checks pass normally and with ASan/UBSan/LSan. They include both
  conversion orders, idempotent solving, unchanged source types, exact binding
  refusal, nonnumeric injection refusal and unchanged exact optional payloads.
- The initial corrected GCC numeric/tagged/concrete matrix passes 19 methods
  in 60.802 seconds; the expanded Clang numeric/tagged matrix passes 14 methods
  in 8.787 seconds.
- Seven numeric-union methods pass on the integrated main base with GCC in
  0.825 seconds. After adding both call-site orders, all seven pass with Clang
  in 1.113 seconds; the changed call-order method and eight existing scalar-join
  methods pass with GCC in 1.598 seconds.
- Positive cases retain int versus float tags on both branch orders, local and
  stack loop backedges, shared call sites in both orders, and optional absence.
  Existing typed-consumer guards remain covered by the adjacent tagged suite.

## My held optional-storage compatibility check

I created a separate checkout from held PR601 pin `1ad605a9` and applied the
numeric-union production commit. The sole textual conflict was the comment
adjacent to my new helper; I retained PR601's scalar-storage comment and source.
This combination is `23a8df53`, with the call-order test follow-up `d92b4ff6`.
No product compiler executable was run.

The combined solver passes 1,269 checks. Seven numeric-union methods and five
PR601 optional-array/projection methods pass together in 4.937 seconds. The
added reversed call-site order control then passes in 0.177 seconds. These are
ordinary focused compatibility checks, not PR601/522 compiler startup acceptance.
The integration checkout remains separate from both owners' source trees.

I retain logs under `/tmp/nanolang-numeric-union-`, including `gcc.log`,
`clang.log`, `final-gcc.log`, `final-clang.log`, `adjacent-gcc.log`,
`solver-asan.log`, `601-build.log`, `601-gates.log` and `601-callorder.log`.
My full native run reports 2,418 passed and four failed. All four failures
compile the standalone map-runtime harness after merged PR613 changed its
runtime text to invoke NVM2C_ABORT. The harness supplies no definition for that
macro; strict C reports an implicit function declaration before execution.
I record this integration prerequisite as
`task_02182f5f162a4bbf9f14803f29d533f3`. I retain the failed full gate and keep
this PR in draft until the repaired harness passes. This is not a passing
full-native claim or an attribution to numeric shape inference. Parent generic
arithmetic and the full release obligations remain separate.
