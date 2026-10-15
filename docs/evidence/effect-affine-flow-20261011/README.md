# My affine effect-flow evidence

I connect retained operation signatures to local handler activation, exact parameter ownership, suspension, lexical return and resumption analysis. I preserve installed-handler and active-arm context at joins. Per-operation escape summaries distinguish locally handled, partial and transitive calls. I also reject HALT with live local owners, which the previous empty-stack check missed.

My state gate passes 469 checks and 510 with allocation faults. The expanded bytecode gate passes 1,557 checks and 2,253 with allocation/visit-limit faults. Positive cases cover resource transfer, lexical return, nested handlers, scalar captures, void/zero-argument operations, contained direct/transitive calls and 64 installed handlers. Negative cases cover dropped/reused arguments, captured-owner reinitialization, wrong argument/result identities, staged/suspended owners, partial/transitive escape, inherited handler pops, handler-context joins, 65 handlers and terminal owner loss. Instruction-fact output remains unchanged on a failed capacity check.

AddressSanitizer and UndefinedBehaviorSanitizer with leak detection pass for the state/bytecode test harnesses with affine_state.c, affine_bytecode.c and ownership_contracts.c instrumented. Other linked objects retain their existing build instrumentation. sanitizer-final.log adds the final void-result fixture to the earlier sanitizer run. Existing scalar-union runtime and ownership-declaration projection gates also pass.

The first negative reuse fixture attempted to store into a captured owner slot, so it was correctly rejected before reaching its intended second move. I retain the initial and diagnostic logs. The corrected fixture stores into a fresh arm local and reaches the use-after-move check; the separate captured-owner test retains its exact destination diagnostic.

Executable ownership verification remains closed for effect instructions, and the tests assert that boundary. Producer emission, effect/global flow, borrowed continuation contracts and runtime cleanup remain required. This evidence establishes analyzer transitions, not full resource-effect execution or 5.1 completion.
