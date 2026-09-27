# I retain exact factory values in callback locals

The original shared-purity function_reference case fails in the preceding
Stage2 compiler. The new effect-counting factory/alias fixture also fails there,
while the C seed passes. I retain both failures.

I reuse the existing exact selector resolver for immutable callback locals.
I check the resolved target signature, then emit the actual initializer before
adding the local name. Factories execute once and retain their returned function
value; aliases load an existing local. I keep the resolved target separately
for supported specialization. Compiler shadows check factory CALL/STORE,
alias LOAD/STORE and signature refusal. Unresolved, mutable, captured and
resource callback profiles are not newly admitted by this repair.

Fresh bootstrap passes, including the changed compiler shadows and both native
stage smoke tests. Native stage binaries differ; this is not canonical bytecode
fixed-point evidence. The complete six-method purity suite and three-method
returned-call suite pass against fresh Stage1 (nine methods, 17.311 seconds).
The same suites pass against fresh Stage2 (nine methods, 40.971 seconds) with:

```
CC=/opt/homebrew/opt/llvm/bin/clang
NANO_CFLAGS=-O1 -fsanitize=address,undefined -fno-sanitize-recover=all -fno-omit-frame-pointer
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1
```

These flags instrument generated native programs; the compiler executables and
external libraries remain ordinary. I separately check generated purity
intrinsics. Original accepted/rejected purity cases and callee-before-argument
ordering assertions remain unchanged.

The complete adjacent generic function-value suite runs 18 methods and retains
14 native translation failures, matching the preceding checkpoint. Their
verified NanoISA products still lack native aggregate/nested-array indirect-call
support. I retain those failures instead of changing accepted programs into
refusals. Resource callbacks, the instrumented compiler-shadow deadline, final
fixed points, complete hosted checks and release documentation remain open.
#522 stays draft.
