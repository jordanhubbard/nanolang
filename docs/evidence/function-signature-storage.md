# My function signature annotation storage

My parsed C function signatures now own parameter and return `TypeInfo` trees
alongside their existing flattened names. Payload copies and interpreter
signature copies preserve those trees independently; teardown releases them.
Nested function returns retain their separate recursive signatures. Nested
function parameters keep their existing parser rejection.

The parser regression checks nested generic array parameters, map parameters,
nested generic returns and chained function returns. It copies the signature,
frees the original AST, then reads and frees the independent copy. The full
parser suite passes with ASan and UBSan; leak detection remains disabled for
existing unrelated metadata leaks. Parser and typechecker suites, the full
native bootstrap, and both existing first-class function execution fixtures
pass, including the integrated parser/typechecker and full native bootstrap
after the generic selected ownership merge. Independent source review found no scoped blocker; the copied annotation
array also checks allocation failure.

Evidence logs are `/tmp/nanolang-signature-storage-tests.log`,
`/tmp/nanolang-signature-storage-integrated.log`,
`/tmp/nanolang-signature-parser-asan.log`,
`/tmp/nanolang-signature-firstclass.log`, and
`/tmp/nanolang-signature-functions-final.log`.

This closes storage prerequisite `task_42ad1f3109374581a3f88f5dc58c049c`.
The parent `task_e05a42e2e09b47cc9c53fa6923eeeaef` remains open: signature comparison,
callable result inference, module metadata serialization, generic native
function-pointer emission and selfhost generic identity still need complete
execution checks. My ordinary `fn()->Box<int>` probe records failures in both
native compiler paths and a NanoVirt publication with later diagnostics; this
storage repair does not claim to fix those behaviors or generic resource
callback transfer.
