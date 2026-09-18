# My single consuming-call runtime prerequisite

MAC `task_4ef48e1066534e639e577c9622445697`, under my open ownership/borrow
parents. I implement the by-value call rule from `AFFINE_TYPES_DESIGN.md`:
the callee owns a moved resource argument and resolves it on every exit.

I retain exactly entry0 and helper1. The consuming helper has one mode-zero
owned record parameter with an exact complete retained layout, supported
scalar/owned locals and one int/bool/u8 result. Entry passes one proven owned
stack value to direct `CALL`. I preserve the existing borrowed-only `CALL_REF`
profile as a separate alternative; I do not mix borrowed and value parameters.

I preflight frame storage before transferring the argument to helper local0.
The caller's moved local remains dead. A fresh callee-local reference activation
and generation identify borrows of the owned parameter or other helper locals.
Native code moves the argument exactly once and clears its previous carrier.
Every reachable return consumes all owned obligations and closes regions.
Assertion, allocation and other terminal errors clean both activations through
the existing lifecycle. Ordinary non-owned CALL behavior remains unchanged.

I require an exact nominal argument/layout match and refuse observations,
borrowed values, moved or held owners, owner results, deeper calls, recursion,
imports, callbacks and reference escape. Existing CALL encoding and ownership
metadata express this contract; I introduce no new wire authority by inference.

I qualify ordinary valid leaf/nested consuming calls, repeated calls, helper
local observations and destruction, all VM lifecycle APIs, assertion and
allocation cleanup, and native sanitizer results. Existing static ownership
refusal and CALL_REF gates remain mandatory. Source admission stays guarded
until this runtime prerequisite is reviewed and merged. I do not execute
malformed modules or historical failed artifacts.
