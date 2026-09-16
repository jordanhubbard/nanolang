# My scalar array compatibility and comparison grouping

I remove two exemptions in my self-hosted checker:

- `types_equal` no longer treats `array<int>` as compatible with every array;
- `apply_return_type_hint` no longer treats an empty scalar element-name
  field as an unknown element kind and overwrites a known type with the hint.

Unknown element kinds still accept contextual types. Recursive and nominal
array-element comparisons remain incomplete and are tracked separately as
`task_3c5d8625cec144ba87efd9695239275a`.

## What the tests exposed

Before the repair, the new source-only gate accepted 42 of 48 invalid scalar
array programs. The cases cross all 12 differing scalar-type pairs with alias
binding, named function arguments, returns and reassignment. The six rejected
cases were not enough to establish array compatibility.
Log: `/tmp/nanolang-array-compat-before.log`.

The first rebuilt bootstrap failed the new `types_equal` shadow because my
native emitter flattened nested equality into a C chain. The assertion was
emitted as `types_equal(...) == i == j`, changing the source's grouping.
Log: `/tmp/nanolang-array-compat-gates.log`.

I retain that shadow and fix the emitter: `gen_binary_op` now parenthesizes
each operand, and general binary emission uses that shared helper. Arithmetic
and logical outer grouping remains; comparison operands retain their AST
grouping without adding redundant outer comparison parentheses solely to
silence or trigger compiler warnings. Dedicated native matrices check nested
equality and relational expressions.

## Final verification on Darwin, 2026-09-15

`make test-selfhost-array-compatibility test-selfhost-map-types
test-selfhost-map-results` rebuilds both stages and passes smoke/no-C-seed
checks, followed by:

- array compatibility: three methods in 15.413 seconds, covering all 48
  rejections, four scalar positive programs and native comparison matrices;
- map typing: two methods in 3.221 seconds;
- scalar map execution: two methods in 50.427 seconds, retaining all 16
  pairings and the exact evaluation-order trace.

Log: `/tmp/nanolang-array-compat-final.log`.

`make test-selfhost-returned-calls` also passes all three methods in
8.575 seconds. Log: `/tmp/nanolang-array-compat-returned.log`.

Each negative case requires the expected structured diagnostic code and
preservation of a prior output artifact. Source-only mode does not invoke C
compilation. Positive programs compile with default shadows and execute
typed empty-array, append and identity behavior for int, float, bool and
string arrays.

Tasks: `task_850ac4914d9b4a9cbc34f7f16dd1902c` and
`task_0f64149182a549b5abf7a6dc2547799e`. This checkpoint does not establish
complete type soundness, all expression semantics or release readiness.
The array compatibility, self-hosted map typing/execution and returned-call
targets are now prerequisites of `test-units`; I have not rerun that complete
suite here, whose other known blockers remain on the roadmap.
