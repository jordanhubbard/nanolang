# My contextual effect/global integration

I follow exact bounded activation and handler stacks from the global entry.
Each control state has a function activation, instruction, active handler stack,
and the possible initialization states of every declared global. Persistent
stacks are interned; only identical control states merge. An ordinary call
creates its own lexical root. A performed arm captures the installing frame's
normal lexical root. Resumption restores its performing continuation; a handler
return unwinds to the captured normal function's caller. Handler installation
creates no speculative arm-execution edge in this global proof.

The pass preserves the shared VM activation/handler depth traps as terminal
paths. Its allocation and 262,144-state storage limits fail closed. Ordinary
branches are conservatively explored in both directions. The existing affine
pass independently checks values, ownership, signatures and local handler
contexts. Instruction facts are staged until both proofs succeed.

My native emitter shares one global store through the effect context and frees
it once at the entry boundary. Every normal/effect activation aliases that
store. Lexical unwind, resumption, traps and allocation failure retain the
existing explicit cleanup protocol.

## Current checks

The raw cases cover initialization on resume, actual perform-time reads,
read-before-initialization refusal, repeated immutable store refusal, lexical
return bypassing unsafe continuation reads, conditional arm initialization,
transitive helper escape, callee-local lexical return and recursive handlers.
Normal bytecode checks pass 1,747 assertions; allocation/capacity-failure checks
pass 2,555. Instrumented analyzer and test code pass ASan/UBSan/leak checks,
including every tested proof-allocation failure and unchanged fact buffers.
Other linked objects retain their existing instrumentation.

All ten combined source cases through both C and freshly built Nano producers pass verification, VM execution and strict
sanitized native C, alongside the existing shared effect-source matrix. The
expanded 26-case runtime suite includes six global cases: scalar initialization,
STRING replacement, lexical return, recursive resumption, assertion failure and
unhandled operation. Native allocation prefixes leave no live allocations;
VM fault checks reuse the same VM after every outcome. Global STRING constants
are retained before execution, so their zero execution-allocation count is not
an assertion that module setup allocates nothing.

I retain the first loop-fixture failure: reinstalling a different handler site
at the join triggered the existing context check before the intended repeated
store check. The corrected fixture keeps one installation around the loop.
I also retain the first eight native source failures on undeclared `globals`;
shared effect-context storage fixes both the signature and lifetime problem.

My fresh Nano compiler-product audit passes all six steps: normal emission with
mandatory shadows (359.16 seconds), verification, native emission, strict C
compilation, startup and the shared effect-source suite. Its pinned inputs stayed
unchanged throughout the audit. I then added only test cases for distinct handler
contexts, two globals, bool/float globals and ordered effectful initializers.
The final ten-case source matrix passes against both producers; the final raw and
sanitized checks pass with those additions. `final-hashes.json` records the final
inputs and retained product artifacts. Production inputs match the audit pins.
All twelve existing global-source methods also pass.

I retain the audit commands and logs under `product/`, final source runs in
`final-c-source.log` and `final-nano-source.log`, and final proof runs in
`final-bytecode.log` and `final-proof-sanitized.log`. I did not rebuild the product
for the later test-only additions. Full 5.1, collection signatures and borrowed
continuations remain open.
