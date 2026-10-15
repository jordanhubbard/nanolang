# Functional callable C AOT

I pass tagged callable-array projections through a checked callable parameter
boundary. I retain the parameter's function representation, constrain the
projected payload to a function shape and preserve the generated tag guard.
I do not admit incompatible scalar payloads as callable parameters.

I distinguish storing a callable identity from invoking its body. I follow
direct calls and resolved, potentially reachable indirect targets from entry
and initialization roots. My bounded constant-flow analysis excludes indirect
calls only when they cannot execute; unknown facts retain refusal. I prune
uninvoked indirect callers transitively when their callees cannot be emitted.

My new `test-native-functional-callbacks` gate executes all nine retained
ordinary callable-operation fixtures and seven generic fixtures through the
C frontend, NanoVM and C AOT with generated-program ASan/UBSan/LSan. I include
an incompatible-parameter refusal that preserves prior output. My dead-call
suite also checks reference-only bodies, reachable unresolved targets and
transitive omission, alongside its existing branch, backedge, mutation and
call-boundary controls.

My final instrumented translator passes 13 focused methods (`sanitized.log`),
including all 16 source fixtures. I instrument `nvm2c.c` and its included
helpers with ASan/UBSan and enable leak detection; other linked translator
components remain uninstrumented. Initial failures are retained in `first.log`
and `pruning-first.log`. The intermediate parameter-only correction is in
`parameter-corrected.log`; the earlier 12-method result is in `focused.log`.

My additional native callable/closure regression passes 19 methods in 12.675
seconds (`closures.log`). My final `make -f Makefile.gnu test-nvm2c
CC=/opt/homebrew/opt/llvm/bin/clang` passes 2,443 C AOT execution checks, 379
callable constraints and 3,098 shape constraints, plus all Python prerequisites
including the 11 dead-call and two functional-callback methods (`full.log`).
The earlier full result before the transitive pruning correction remains in
`intermediate-full.log`. This checkpoint does not refresh
self-hosted stages or establish exact-candidate Linux/Darwin qualification.
