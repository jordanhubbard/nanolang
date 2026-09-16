# Explicitly tagged record arrays

The compiler's first unsupported array construction was function 18,
`tokenize_string`: `ARR_NEW 8` explicitly requests struct elements. I now
recognize that tag as a record-array representation in both classification and
C emission, including construction without an immediate local store. Empty
array field facts begin unknown rather than inventing integer fields.

I retain the existing legacy inference path and its tests. I also reject
appending a known incompatible record-field representation rather than
overwriting the earlier array facts. This does not implement nested arrays,
arbitrary recursive aggregates, nominal layout proof, or unbounded record
storage. The existing AOT limits of eight scalar/string record fields and
256 record-array entries remain.

## Verification

`make -j1 test-nvm2c` passes 922 checks on Darwin. New generated executables
exercise explicitly tagged empty arrays, mixed integer/string records through
local storage, and direct stack construction/push/get. Negative cases reject
scalar insertion, unsupported nested-array construction and mixed record-field
representations. Existing legacy record-array tests remain passing.
`git diff --check` passes.

`make -j1 test-one-ir-compiler` advances through `tokenize_string` and now stops
at function 20, `parser_init_ast_lists`, with an operand-stack overflow in the
AOT translator. Full compiler acceptance remains unfinished under MAC
`task_419c47bdc8fc42e4b52eb6af1a0e9a71`; the array-shape roadmap item remains open.
