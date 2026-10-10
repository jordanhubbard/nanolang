# My native record-name and callback fixes

I reproduce the unchanged `tests/nanoisa/fixtures/record_arrays.nano` failing
because its ordinary `NSType` record collides with my internal schema typedef.
`original-refusal.log` retains that failure. I distinguish non-extern source
records from runtime-provided typedefs using the active emission environment,
restore that environment after emission, and use one naming function for
record declarations, generic arguments and array operations. Extern records
retain their existing ABI spelling. I remove the old test workaround that
renamed `NSType` to `ValueType`.

My new generic-record fixture also finds a real checker omission: only
identifier arguments supplied names to generic specialization, so record
literals emitted `identity_unknown(int64_t)`. I use the shared expression
identity lookup and test literal, local and ordinary returned-record arguments.
`generic-literal-refusal.log` preserves the original failure.

My callback fixture reaches another independent failure before C publication:
C-seed shadows reject record-array filter/map/reduce. I copy records with the
same representation used by array access and append, preserve declared record
map outputs, and handle empty filter inputs without relying on calloc(0).
`callback-shadow-refusal.log` retains that failure; I do not disable shadows.

`record-name-tests.log` passes six methods: the original nested-record fixture,
six runtime-like name families with array get/set/pop, imported records,
external schema ABI, generic record arguments/results and actual callback
execution on static, dynamic and empty arrays. `native-neighbors.log` passes
four existing record-array methods, the first five new methods, StringBuilder
unit checks and two assertion-emission controls. The standalone primitive
generic fixture also compiles and executes successfully.

I initially built an isolated compiler prototype while the earlier byte-read
bootstrap was live, leaving its 1,204 captured source inputs unchanged. That
bootstrap completed with equal Stage 1/2 raw modules before I applied these
source changes. Its receipt is evidence for byte-read emission; it does not
qualify this later revision. Exact-candidate bootstrap/platform gates and the
remaining full 5.1 host/module/aggregate requirements stay open.

## My final evaluator and functional-array controls

`evaluator-functional-neighbors.log` passes the complete evaluator unit suite
and builds the real self-hosted emitter from the changed C compiler. Seven
functional-array methods then pass, including VM/C AOT sanitizer execution,
callback ownership, scalar type changes, source order and empty outputs.
The eighth correctly refuses the shadowed callback and preserves the prior
output, but its diagnostic assertion still expects the obsolete word `indirect`.
I retain the failure and require the current exact diagnostic instead.

`functional-refusal-corrected.log` reruns that exact method's eight signature
refusals and shadowed-callback refusal, retaining exit-status and output
preservation checks. I invoke the method directly because it does not use the
expensive class-level shadow-driver fixture. The other seven methods already
passed with that real fixture; I do not describe the original full invocation
as green.
