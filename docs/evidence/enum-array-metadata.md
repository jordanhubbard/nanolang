# I retain scalar enum-array metadata

My parser retains a named array element as a nominal annotation. I now resolve
that name against enum declarations before checking literal annotations,
selecting array access types, or choosing native storage helpers. Ordinary
record annotations retain their record representation. My native emitter uses
integer dynamic storage for enum literals and appends.

My regression executes declared enum locals, direct `at` and `array_get`,
function parameters and returns, record fields, globals, count/fill construction,
mutation, append and empty field context. Eight exact function bytecode
comparisons pass between the C seed and self-hosted emitter. The C native
program, both VM modules and both strict C11 AOT programs execute successfully.
Four C-seed/NanoVirt wrong-kind cases reject bool and record literals while
preserving the prior artifact.

The full emitter gate passes 86 checks and 46 integration methods. Adjacent
record-literal and nominal-array gates pass. A fresh three-stage bootstrap and
typechecker suite also pass; the final equivalent metadata-lookup cleanup passes
the full emitter and adjacent record gates again.

I do not infer new enum nominal equality rules, array-pop lowering or the full
compiler bytecode fixed point from these scalar metadata checks.

Evidence: `/tmp/nanolang-enum-array-before.log`,
`/tmp/nanolang-enum-array-final-gates.log`, and
`/tmp/nanolang-enum-array-frontend-gates.log`.
