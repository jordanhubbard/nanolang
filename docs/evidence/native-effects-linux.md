# My native effect execution checks

I replace native perform placeholders with synchronous dynamic dispatch. I lift
handler and handled bodies into C functions, pass lexical captures by address,
and share handler state across imported compilation units. Perform arguments
run in source order before handler parameters enter scope.

A handler's final expression resumes the perform. An explicit return leaves the
function that installed the handler, including when a helper performed the
operation. I place `setjmp` in a separate runner, so modified captured variables
remain ordinary caller storage. Before a nonlocal exit I release abandoned owned
locals while their stack storage is live. I retain an owned captured return
before its installing scope releases the old reference.

I reject nonlocal handler returns across a native foreign callback activation
with a diagnostic and abort. A handler that resumes normally can run across that
boundary. I do not claim general foreign-stack unwinding or asynchronous
continuations.

On Linux ARM64 I exercised:

- The three native halves of `tests/test_effect_execution.py`: observed handler
  state, lexical return versus resumption, and ordered multiple/zero arguments.
- Ten `tests/test_native_effect_execution.py` cases: mutable captures, nested
  handler installation/restoration, imported dispatch, frame removal, accepted
  and rejected foreign callback paths, and owned cleanup/return transfer.
- One hundred nonlocal exits in `tests/test_native_effect_runtime.c`, with one
  finalizer per abandoned allocation and empty handler/cleanup stacks afterward.
- The same ten Python cases with generated native programs and runtime compiled
  using AddressSanitizer and UndefinedBehaviorSanitizer. Leak detection was
  disabled for this suite; it is not a leak-freedom claim.
- Parser, transpiler and typechecker unit gates, including the assertion-literal
  execution tests.

These tests do not establish equivalence for every effect signature or backend.
My 5.0 effect roadmap remains open until the combined native/VM and release
checks pass.
