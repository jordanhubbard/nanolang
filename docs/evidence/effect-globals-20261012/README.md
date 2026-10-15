# My effect/global flow prerequisite

I track the full combined implementation under #976. I retain five source
programs through both current producers: handler mutation, helper dispatch,
lexical return, effectful global initialization, and retained STRING mutation.
All ten baseline runs refuse publication and preserve the prior artifact.
The C producer rejects the combination before emission; the Nano producer
reaches the affine verifier's explicit combined-flow refusal.

I introduce a shared finite global transfer algebra and use its existing
precondition/composition operations in ordinary affine global analysis. A
transfer records permitted entry initialization states, ordinary/resumed exits,
and lexical-return exits separately. Sequence applies the second precondition
only to first-transfer ordinary exits. Earlier lexical returns bypass it.
Alternative control paths intersect safety preconditions and union outcomes.

My exhaustive oracle enumerates all 1,048,576 pairs of encoded transfers,
including divergent paths and refused inputs, comparing each individual path
with the compact representation. Explicit controls check initializer/read,
repeated immutable store, mutable store, conditional initialization and
lexical returns bypassing later operations. The initial oracle overwrote an
already collected lexical-return outcome during enumeration; I retain that
failure and correct accumulation without changing any production expectation.

Normal and LLVM ASan/UBSan/leak runs pass 20,971,528 assertions each. Existing
affine bytecode regressions pass 1,557 checks, and allocation-fault regressions
pass 2,253. The transfer algebra allocates no memory. This is tested finite
transfer behavior, not a proof of complete effect/global execution.

## Remaining integration

I must bind handler summaries to the exact lexical installation/activation,
apply global preconditions at the actual perform/call state rather than the
handler-install state, and propagate distinct resumption and lexical-return
relations through nested handlers, escaping direct calls, joins and loops.
A handler's lexical return must unwind to its enclosing activation, not the
performing helper's continuation. Recursive handlers require a bounded,
monotone fixed point. I must retain transactional instruction facts and
allocation-failure behavior, then qualify the five source cases plus negative
initialization/store cases through both producers, VM and native execution.

The verifier refusal and C producer refusal are unchanged. No new combined
program is admitted by this checkpoint; the full roadmap item remains open.

My later [contextual integration](integration/README.md) qualifies handler/call
flow and paired source execution with a fresh compiler product.
