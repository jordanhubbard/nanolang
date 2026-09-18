# Lexical scalar declarations in borrowed source

I extend my bounded borrowed-source control flow with int/bool declarations.
My prerequisite is the scalar definite-initialization meet from PR624.

I evaluate an initializer before introducing its new lexical name. Each local
receives a distinct physical slot and exact scalar ownership descriptor. At a
branch or loop body exit I close its advisory name interval and restore lookup
of outer bindings; I retain the slot metadata and do not invent initialization
on paths which skipped the declaration. A loop iteration executes its stores
again. Compiler temporaries remain unnamed. Selected shadows use these same
rules, and I refuse publication if any selected shadow cannot be lowered.

I retain exact owner/reference/region joins. My
[resource-path extension](NANOISA_SOURCE_RESOURCE_PATHS.md) separately admits
explicitly consumed resource declarations, patterns and whole-owner moves.
I still refuse break and continue inside control flow. My
[return-path extension](NANOISA_SOURCE_BORROW_FALLTHROUGH.md) separately admits
bounded scalar early returns after explicit owner consumption. I do not introduce implicit resource disposal. Existing depth and local
limits remain in force.

My acceptance requires both source producers, canonical Stage1/Stage2,
matching code/layout/ownership/name metadata and selected-shadow modules.
I test both branch outcomes, zero and multiple loop iterations, nested scopes,
initializer access to an outer binding, restoration and out-of-scope refusal.
I execute verified modules in the VM and sanitized native output, and strip
advisory names without changing execution.
