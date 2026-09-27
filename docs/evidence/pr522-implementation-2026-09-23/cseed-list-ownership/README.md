# My C-seed schema-list ownership

I wrap schema-list constructors in generated native C, preserving their raw
pointer ABI and adopting each result into the shared process owner. Explicit
free unregisters the pointer before calling the original runtime destructor.
Nested parser lists are independent allocations with shallow destructors;
retiring an outer list does not invalidate an independently retained child.
Returned values and cross-module aliases keep their process lifetime. My
private C helper names cannot collide with ordinary `nl_` user-function names.
This is process-exit cleanup, not lexical collection or bounded retention.

The owning fixture exposed a separate checker/interpreter defect: splitting
at the last underscore treated `with_capacity` as part of the element name.
I retain that failure. A shared separator now preserves compound operations;
constructor checking requires the exact arity and INT capacity. I check the
language INT range before narrowing to the runtime's native int. Unsupported
custom-list capacity constructors refuse before output publication; this does
not introduce a custom-list backend implementation or establish canonical
backend parity. I track the C-seed correction as
`task_5db32ae1e7844a3f8882e5358e4772cb`.

My targeted tests cover nested storage, growth from initial capacity one,
retained aliases after freeing the outer container, explicit child free,
returned lists, a separate module, a user-function collision control, zero
capacity growth, wrong constructor argument types/counts, and negative/oversized
capacity refusals in native execution and shadows. Previous-output sentinels
remain unchanged on rejected compilation. All ownership products retain the
existing ASan/UBSan/leak/UAR settings and stricter conservative-root controls.

Final-source verification passes all six targeted methods in 52.348 seconds,
including the private-name collision case. The full ordinary command
`make test-typechecker test-runtime-lists test-transpiler` passes its fresh
bootstrap, all typechecker tests, 33 AST plus six non-AST runtime-list types,
StringBuilder checks and both assertion-literal methods.

The rebuilt final-source instrumented Stage 1 compiler still fails its original
hello smoke with 4,100 bytes in four allocations, all owned physical-path
results. Its preceding checkpoint reported 18,724 bytes in 92 allocations:
this repair removes 14,624 bytes and 88 parser-list allocations. I keep the
intermediate helper-name checkpoint and fixture/build corrections separately.
This is not full instrumented bootstrap or installed-package qualification.
The remaining path-result leak stays under
`task_02204077a4d34d4aa5ce20ef7054e113`.
