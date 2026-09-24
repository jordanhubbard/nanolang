# Compiler metadata lifetimes

I continue PR #522 from `d74ae8a6620d67e4176eb70b7f6fa2e04a0e9f56`.
I retain the original hosted leak failures in the preceding evidence directory.
This directory records local reproduction, intermediate failures and corrected
checks. `sources.json` identifies the changed sources; `instrumentation.json`
verifies ASan and UBSan symbols in fresh codegen, environment, typechecker and
transpiler objects. I compile those objects with Homebrew LLVM, `-O1`, both
sanitizers, frame pointers and no sanitizer recovery.

## Registration and checker storage

The unchanged introspection gate reproduces the hosted 26-byte qualified-alias
leak on Darwin. I own qualified function names explicitly and release them on
both successful and refused code generation. Ordinary AST/module names remain
borrowed. Foreign import entries now borrow the stable strings already present
in the output module instead of allocating unowned copies.

The existing generated-list metadata test reproduces 385 leaked bytes in 14
allocations with leak detection enabled. I retain generated function names,
parameters and result names in the existing environment ownership registry,
independently of mutable function-table slots.

The full NanoVirt gate initially reports 4,495 leaked bytes in 125 allocations.
I release owned effect declarations, discard handler-local symbol names and
values when restoring their previous scope, and free tuple element-type caches
before replacement. Borrowed record bindings retain the owner's value. The
new scope test checks repeated shadowing, alias release and indexed slot reuse;
the existing borrow test checks that popping aliases leaves the owner alive.

The large canonical-match shadow-driver setup reproduces 277,934 leaked bytes
in 15,308 allocations. After generated-list cleanup it reports 87,384 bytes in
8,224 allocations. The remaining stacks identify copied empty struct-parameter
placeholders, enum module names, replaced list-result type names, and foreign
module metadata when its header array is absent. I correct those owners while
retaining borrowed AST annotations and unchanged source behavior.

## Test portability and strict GCC

My empty-export introspection trace compares `/var/...` with `/private/var/...`.
The resolver already canonicalizes imports. I compare against the exact resolved
path, add an explicit symlink-import control, and select the requested C compiler
for generated sanitizer programs. All five introspection methods pass with
instrumented NanoVirt, ASan/UBSan and leak detection. Rejection cases also require
absence of sanitizer diagnostics and preserve their prior output artifact.

Hosted Linux x64 and ARM at the preceding head both stop at GCC's
`-Werror=format-truncation` in my new nested-array fixture. The isolated GCC
control reproduces the warning and its maximum 8,223-byte output against an
8,192-byte buffer. I size the buffer from both bounded inputs plus the fixed
prefix. The identical strict GCC compile then passes without suppression.

## Qualification boundaries

The complete NanoVirt gate passes all 90 tests with leak detection enabled,
and the generated-list metadata gate passes under the same instrumentation.
I retain intermediate local 10-second shadow timeouts; the corrected qualification
uses CI's existing `NANO_SHADOW_TIMEOUT_SECONDS=60`, without changing production
or CI deadlines. The complete match run then reaches its tests without setup
leaks but two expected-invariant controls select Apple's C runtime, which rejects
leak detection before execution. I retain those terminals and select the test's
existing `NANO_NATIVE_TEST_CC` Homebrew override for the final run.

My final canonical-match gate passes all ten methods with the selected Homebrew
compiler and leak detection. The owning environment-scope target passes 109 C
assertions and all ten lexical-scope methods. Its prerequisite also rebuilds both
native stages and passes installed-compiler smoke checks with the instrumented
build configuration.

The original functional-array gate now completes shadow-driver setup without
leaks, but one of nine methods fails: `nl_functions_array_param.nano` is refused
by native translation with `STORE_LOCAL: expected string value`. I track that
remaining backend defect as `task_f78020ad5c124c36bc26175241bc2539`; I do not count
this gate as passing.

The separate Linux sanitizer preparation terminates at the unchanged 600-second
transpiler component deadline. Its isolated source copy is the preceding d74
checkpoint, not these lifetime changes. Its existing bootstrap sentinel caused
Make to include native bootstrap before component compilation; this is not the
fresh-checkout hosted sequence. I retain its complete terminal separately.
Final hosted/platform qualification, canonical bytecode fixed points and release
documentation remain open. These local checks do not establish all of #522.
