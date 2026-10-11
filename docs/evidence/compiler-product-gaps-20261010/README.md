# Compiler-product implementation gaps

I audit both existing compiler-product routes at 73571dba5 and retain their
commands, terminal failures and unchanged source inventories. My C-frontend
NanoISA compiler translates to strict native C and starts, but its acceptance
suite exposes two C-frontend nested-return closure refusals. My other route
fails while compiling `service_inputs.nano`: generated C includes
`file_source_inputs.h` without the imported module's include directory.

My first acceptance run also used Apple Clang with LeakSanitizer enabled,
which that runtime rejects. I retain those failures in `initial-04.stderr`.
With Homebrew Clang, the unchanged ten-method acceptance suite isolates the
two real C-frontend signature failures (`adjacency-first.log`). My named
returned-function control reproduces the same missing annotation
(`named-first.log`). The self-hosted producer passes these source controls.

I normalize a declared function's complete returned-callable annotation when
constructing its value signature. Both annotation representations own complete
copies; I preserve exact signature comparison. My new incompatible named and
anonymous return controls fail before replacing prior module output.

I also pass imported foreign compiler flags into pure Nano module object
compilation. I select build records by the isolated module's actual import
closure, retaining quoted include paths and existing overflow refusal. A
small import with a spaced include directory reproduces the original missing
header before the correction (`headers-first.log`).

My corrected capture/record/scope suite passes twelve methods
(`adjacency-corrected.log`). All five module-generation methods pass
(`headers-corrected.log`), including transitive headers, the pure Nano import,
long link closures, oversized flag refusal and shared foreign generations.
My 45 selected callable/generic regressions pass with nine compiler components
instrumented by ASan/UBSan (`sanitized.log`, leak detection disabled). Environment
checks pass 60 C assertions and ten lexical-scope methods; evaluator units
also pass (`units.log`).

My retained audit runner keeps the original explicit compiler-stage deadlines
and all test assertions. Its unspecified helper deadline is 120 seconds
(the original helper defaults to 180); it does not raise a shadow deadline.
My corrected first compiler-product route passes. Its native compiler
compiles its full source to a verified NanoISA module in 742.000 seconds,
passes the twelve-method acceptance suite, and publishes matching Hello
programs through native and NanoVM execution. The independent C-seed native
compiler route reaches shadow execution but fails at its unchanged ten-second
deadline (`corrected-shadow-failure.stderr`). The two-test audit takes 944.459
seconds and finishes with one pass and one failure; its source inventories
remain unchanged (`corrected-manifest.json`).

I also translate the freshly self-hosted `rebuilt-compiler.nvm` from the first
route to strict C11 and execute the resulting compiler. Its twelve-method
capture/record/scope suite passes, and its emitted Hello module executes
through both NanoVM and native C AOT (`selfhost-followthrough-manifest.json`).
This exercises the actual self-hosted compiler artifact; it does not replace
or erase the independent C-seed shadow failure.
I do not claim refreshed installed generations or exact-candidate Linux/Darwin
release qualification from these local implementation checks.

My default-deadline traced C-seed build reproduces the timeout
(`full-compiler-shadow-trace.log`). I complete 678 shadows with no assertion
failures, accounting for 9.553542 seconds, then stop during
`nb_owner_array_fixture`. Completed parser, typechecker and borrow-emitter
shadows account for 2.323801, 3.126011 and 3.740021 seconds respectively.
This establishes cumulative suite exhaustion; it does not establish the
remaining shadows would pass, or identify the interpreter performance cause.
I retain the full selected suite and unchanged deadline for correction.
