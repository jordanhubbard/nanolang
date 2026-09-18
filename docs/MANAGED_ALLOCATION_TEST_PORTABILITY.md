# My portable managed allocation controls

I track task_a29232f397fd44c688734e090bead9f9 after merged PR735. Static
inspection identifies thirteen remaining GNU malloc-wrap call sites in my
managed string, trim, split, replacement, mutable-array, array-copy, format,
primitive-format and case tests. Apple ld rejects that option. I do not execute
those unchanged linker paths on Darwin.

I factor PR735's reviewed test-only allocation redirection into the shared
native harness through an explicit optional allocation-control argument. When
selected, I copy the freshly generated native LLVM text, require an actual
malloc declaration and call, and replace only the exact malloc symbol with a
private test callback before the existing ASan instrumentation and object link.
The original generated IR stays unchanged. The separate C callback calls
ordinary libc malloc; harness/sanitizer allocations do not consume the module
budget. Ordinary harness calls remain unchanged.

I migrate each affected caller to the explicit option and ordinary C callback,
removing GNU linker wrapping. I also migrate the already portable graph fixture
to the shared implementation to avoid duplicate allocation-redirection rules.
I retain every budget, expected error, live-object/byte count, successful later
entry, alias/global/disposal observation, optimization flag and sanitizer pass.
No production runtime, compiler, LLVM output or Wasm path changes.

I freeze code before qualification. On Linux and the authorized Darwin peer,
I build fresh tools on the selected source and run the full existing generated
managed corpus with fail-fast reporting, plus the private record target tests
introduced by737. Every original inner timeout remains; each build or suite
has a900-second outer bound and stops at the first failure. I record exact
source/tool hashes, versions and logs, distinguishing any unsupported or
unexecuted route from a pass. I do not replay historical crash artifacts.

This completes a test-portability prerequisite, not full nominal/aggregate,
product or release acceptance. Full ordinary record authority and lowering,
owned results/source admission and the product candidate remain separately open.

My first Darwin qualification at494f passed17 tests, then exposed one additional binary64-format GNU wrapper omitted from the initial inventory. I retain that failed run and track0b732 before replacing only this final wrapper interface. All allocation budgets and expected results remain unchanged; I repeat the full71-method qualification on the corrected frozen pin.
