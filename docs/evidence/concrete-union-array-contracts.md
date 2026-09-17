# I resolve concrete union array payload contracts

My fixed payload check cannot treat the formal `T` in `Box<T>` as a concrete
record. I now pass the enclosing declaration into a separate constructor check,
copy and substitute retained payload annotations, and compare nominal record
identity after substitution. Temporary annotation copies are freed after each
check; I do not attach borrowed context to the expression AST.

I check locals, globals, assignments, ordinary and qualified call arguments,
function returns, nested union constructors and nested array literals or typed
aliases. Empty arrays preserve their context. An unrelated record named `T`
does not replace a substituted concrete argument.

Two regression methods cover sixteen rejecting C-seed/NanoVirt decisions and
six matching native executions. Rejections retain prior output artifacts. The
five C union literal-context methods from PR424 also pass, including its formerly
blocked generic wrong-record case. My fixed nominal array regression remains
green. A fresh three-stage bootstrap and typechecker suite pass; the final
nested-alias addition also passes both focused contract suites.

Correct generic globals, inline arguments, mutable assignments and nested union
constructors exposed a separate native emission defect: their concrete type is
lost before C compilation. MAC `task_633f2402ec5944cfba0911a56a9f4eb1` retains
those positive controls. This check rejects wrong payloads before emission; it
does not claim native support for those existing failing contexts, generic scalar
payload equivalence, resource transfer or the bytecode bootstrap fixed point.

Logs: `/tmp/nanolang-concrete-union-gates.log`,
`/tmp/nanolang-concrete-union-nested-gates.log`, and
`/tmp/nanolang-concrete-union-integration.log`.
