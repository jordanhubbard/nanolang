# Self-hosted map prerequisite

I added `make test-selfhost-map-results` to execute the unchanged map result
matrix on Stage2. It selects all 16 scalar input/result pairs, with named,
variable and returned transforms and append-after-empty checks.

On 2026-09-15, both compiler stages rebuild and bootstrap smoke checks pass.
The map gate then exits 2 with 16 failures. Each fixture fails while parsing
the returned-function call in the `choose` shadow, before map lowering:
`((choose) 7)` reports an unexpected `7`; other scalar arguments fail at the
same position. I do not count these failures as a map emitter test.

`ASTCall.function` currently stores an identifier index. The checker and
emitter retrieve that identifier rather than a general callee expression.
The prerequisite must carry expression callees through parser, generated AST
transport, checking and emission, with nested calls and evaluation-order tests.
I did not remove returned callbacks from the matrix to bypass this boundary.

Log: `/tmp/nanolang-map-selfhost-gate.log`. MAC task
`task_256337a9977f43b2baee7b26ebd66bc7` was reported running under `agent_rocky`
when inspected. That claim is not completion evidence. Map parity remains
open under `task_75b340982b6cf797f29b38c1a188aab3`.
