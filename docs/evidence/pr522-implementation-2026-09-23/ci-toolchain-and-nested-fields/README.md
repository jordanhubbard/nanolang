# Sanitizer toolchain and nested array fields

I continue PR #522 from `530fb29c1aa9746477eb11b34bab1b4e54f9f35a`.
My `sources.json` records the implementation and test hashes for this checkpoint.
I keep final hosted acceptance and canonical bytecode fixed points open.

## Sanitizer selection

I select distribution Clang with its matching compiler runtime for sanitizer
Make phases, generated products and C harnesses. I retain every partition,
original test target, shadow deadline and ordinary platform compiler selection.
The preparation phase builds components, then explicitly runs `bootstrap3`.
The component `build` and native-compiler `bootstrap3` targets check different
artifacts; I now explicitly refresh both native compiler stages before tests.

The required preflight compiles one bounded canary, executes 100,000 valid calls,
and requires specific ASan use-after-return and UBSan signed-overflow diagnoses.
A crash alone is not a pass. I retain actual Linux Clang success and GCC 13.3.0
failure in `linux-canary-report/` and `gcc-canary-report/`. For the GCC control
I remove only Clang's `--rtlib=compiler-rt` linker option. GCC passes the valid
and overflow cases but misses use-after-return. Ten partition harness methods
pass, including rejection of a missing, failed or skipped canary step.
`linux-sanitize.log.gz` records the selected-toolchain `make sanitize` pass.
These are local measurements, not completed hosted partitions.

## Native compiler memory

The first 8 GB Linux run passes the instrumented C seed's shadows and Stage 1
hello smoke, then the ordinary native Stage 1 is OOM-killed during self-compilation.
`bootstrap-kernel.log.gz` records 7,823,988 KB anonymous RSS. The bounded debugger
samples and real GC statistics in `native-memory/` repeatedly locate allocation
in `lookup_field_type` rebuilding `build_field_metadata_index` for each lookup.
The driver stops this separate diagnostic at its stated sample/memory bound.

I replace that repeated map construction with a reverse metadata scan. I retain
last-definition precedence, the exact flattened dotted-key collision behavior,
the existing limited alias capitalization and named array-element metadata.
An initial shadow wrongly assumed arbitrary capitalization; I retain its failure
and corrected fixture. The first reverse-scan prototype reaches external Clang
but still exhausts the 8 GB VM with both processes live; its patch, hashes and
kernel evidence remain separate from my final implementation.

I qualify the final metadata scan on a dedicated 16 GB, two-CPU Linux VM, matching
standard public Linux hosted-runner memory. This is not evidence of 8 GB support.
The Linux run uses `bootstrap3 build`, before my final phase-order correction,
and predates the nested setter and native translator changes below. Its bootstrap
passes; the additional component build fails its unchanged 600-second transpiler
deadline. Parser and typechecker components pass. Total elapsed time is 30:49.54,
with reported maximum RSS 6,301,820 KB. The orphaned Clang processes from that
expired compilation were terminated before final-source rebuilding. I retain
this failure; a successful bootstrap is not a complete pipeline pass.
The native stages in that run are not claimed to be fully instrumented: the C
seed and retained runtime objects are instrumented, and the generated component
compile command carries sanitizer flags.

## Nested array replacement

The unchanged source fixture fails both native stages and also the saved Linux
Stage 1 from before the metadata optimization. My opcode trace records an empty
integer array receiving a string. I preserve the receiver, index and replacement
expected types in source lowering. The corrected VM trace passes its assertions;
that correction then exposes native C translation refusals.

I box concrete child arrays through tagged record fields for both replacement and
append. At array return, I constrain the payload of optional storage instead of
equating its wrapper with an array. Checked native tag/presence guards remain.
Eight reduced positive cases cover four scalar child types in both declaration
orders, writes visible through aliases and returned record fields. Two negative
controls initially exposed accepted incompatible child types; I retain that
failure in `nested-reduced-before.log.gz` and add the missing exact recursive
write constraint. The shape solver itself is unchanged.

The final 45-method compatibility gate passes after fresh bootstrap and the
negative write constraint, across the C seed and both native stages with
generated-product ASan/UBSan and leak detection (`darwin-source-final.log.gz`).
Compiler executables themselves remain ordinary in that source run.
Both final translator gates pass 2,484 assertions: ordinary and fresh private
ASan/UBSan builds, including instrumented generated C. The sanitizer driver
verifies instrumented translator and shape objects. This owning gate keeps its
existing leak-detection exclusion; source compatibility checks separately enable
leak detection. Deliberate guard abort messages are expected negative controls.
Final-source ordinary Darwin `bootstrap3` passes both stages and installed
compiler smoke checks, including operation with the C seed removed. Native binary
inequality remains explicitly distinct from the canonical bytecode fixed-point
requirement.

## Evidence boundaries

My logs distinguish failed prototypes, intermediate passes and final checks.
`nested-backend-build.log.gz` records a no-op attempt to build `bin/nvm2c`; the
owning `make nvm2c` target in `nested-backend-build-corrected.log.gz` actually
rebuilds the translator. I do not count the former as qualification.
I keep #522 draft until complete hosted platforms and sanitizer partitions,
final-source canonical `.nvm` fixed points and release documentation qualify.

## Latest hosted evidence

I retain failed-job output from run `35957468323` at the preceding pushed head.
All three platform jobs and coverage reproduce the unchanged nested-array setter
failure corrected here. The sanitizer results also contain two separate leak
failures and two timeout cases. `hosted-followups.json` records their MAC tasks;
my roadmap retains the exact gates. I do not attribute these failures to the
GCC problem without reproduction. The final Clang pipeline and complete hosted
acceptance remain unfinished.

My nested-array MAC description records the verified correction. Direct completion
is rejected because the task is already in `failed`; I retain that lifecycle
response and do not claim the ledger task is closed.
