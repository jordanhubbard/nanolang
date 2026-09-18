# My ownership-preserving source control flow

MAC `task_89c4d7131a664f5a8aadbbbbc4b5389e`.

I extend completed nested source-borrow task91cb under affine parent
task_ed70242ac4d83be7b2327da7ece387ad and borrow parent
task_71821d84befc46e198795122c1112a27. I consume the existing owned verifier's
exact branch and loop joins without changing runtime authority.

I admit if/else and while over existing initialized int/bool locals, scalar
assignments, assertions, nested control flow and the existing exact borrowed
helper calls/field operations. Conditions must be Boolean and consume their
operand. Each borrow/call region ends before a branch or backedge; actual
owned roots and formal reference origins remain unchanged. Branches execute
only the selected arm; loops re-evaluate conditions, including borrowed calls,
in source order.

This original slice refused any let. My [lexical scalar extension](NANOISA_SOURCE_BORROW_LEXICAL_SCALARS.md) now admits bounded int/bool declarations after the scalar initialization meet. I still refuse resource construction/destructuring/move,
return, break or continue inside control-flow bodies. My original boundary required scalar locals before control flow; the extension
uses explicit definite initialization rather than inserting dummy stores. Nested control bodies
have a bounded depth of 32. Existing top-level owner construction, moves and
complete disposal remain unchanged. No implicit drops.

Both C and selfhost producers emit actual conditional branches and backedges
with identical code, ownership/layout/path contracts and advisory names.
Selected shadows use the same lowering and must all succeed or publication
refuses. Ordinary positive tests cover both arms, zero/multiple loop
iterations, nested branches/loops, helper-body control flow, ordered side
effects, repeated shared/exclusive nested-path calls and post-loop complete
consumption. Refusal controls preserve existing ownership/purity/nominal
boundaries plus unsupported local initialization and early-exit forms. VM and
sanitized native must execute the exact same artifacts. Broader path-sensitive
owner moves and local lifetimes remain separate required ownership-flow work.
