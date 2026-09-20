# I qualify my original managed-string acceptance

I record task_5792220dc3654ddcbe7e47ec0253f8ea under original parent
`task_51da49b39230468784da3481b893563b` before changing any fixture. My base is
actual grant merge dc83a0c92432534226d50b422a9d0b8e78657f79. This is a test and
acceptance reconciliation, not a new opcode or executable admission.

My original [contract](NANOISA_MANAGED_STRINGS.md#my-acceptance-evidence) promises
VM, native LLVM before and after optimization, and Wasm comparisons. Core,
frame cleanup, concat, substring and portable conversions are merged. The
Darwin parser child7ba is also complete through its retained evidence; its old
open wording is historical. Aggregate/cycle488, host linkage2d2 and full
applicable-language coverage remain separate requirements.

Static review of `tests/test_llvm_managed_strings.py` shows LLVM verification,
explicit ASan instrumentation and llc object generation. The C harness is built
with -O1, but that does not apply a LLVM optimization pipeline to the already
emitted module. I add a test-only selector for the exact `default<O2>` pipeline
to the common emitted managed-string fixture. The absent selector keeps the
existing route. Unsupported selector values fail setup instead of silently
selecting another mode. I retain the original and optimized native/wasm32 IR,
verify both, and run the same unchanged behavioral assertions. Structural
checks of emitted cleanup/traps apply to original IR before optimization; they
are not claims about an optimizer's symbol or call-count preservation.

I apply optimization before marking/instrumenting the selected native IR for
ASan. I keep the explicit ASan pass and checks for actual instrumentation.
Native allocation controls continue to replace only generated-module malloc;
the instrumented C harness and linked engine scope remain accurately distinct.
Wasm stays import-free and uses the same runtime exports, bounded memory and
actual Node/Wasmtime controls. No expected failure becomes a success, and no
publication sentinel, lifetime counter or page bound is removed.

I qualify the original11 emitted string methods and the existing decimal and
portable float-format conversion methods in both modes. The unchanged core,
package and binary64 parser/formatter acceptance supplies allocator, growth,
reference-bit and production-mode coverage. Before choosing the exact runner
I inspect those modules and all prerequisites; shared helper inheritance must
not accidentally omit a promised method. Existing scalar/literal profile and
verifier refusal neighbors remain part of the original contract.

I use fresh isolated Linux and Darwin tools with explicit supported Clang/LLVM
selection. Every command retains its actual status, bounded process-group
cleanup and output before assertions. Temporary sources, IR, Wasm, binaries
and module-generated objects remain archived; source/tool/provider maps
distinguish immutable inputs from generated products. Failed terminals are
preserved without replay. The source and fixture changes receive independent
review before execution. No complete managed-string or parent closure follows
until an original-criterion evidence matrix supports every claimed boundary.

The unresolved historical evaluator incident791a retains its own evidence.
It does not establish a current managed translator defect, nor does a passing
new gate retrospectively explain it. Full product/fixed-point and release
acceptance remain required.
