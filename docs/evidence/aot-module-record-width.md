# Module-derived aggregate width

I remove the eight-field aggregate limit. I decode construction instructions
before classification and derive one field width from the widest aggregate
in the module, with one slot for modules containing only empty aggregates.
The bytecode's 16-bit field count bounds this width; there is no new arbitrary
field-count cutoff.

Classifier vectors use function-scoped owned allocations. Published vectors
are immutable; branch merges clone before changing facts. Emitter record and
record-array tables remain separate and use checked dynamic field storage.
Argument, local and result facts use the same module-derived stride. Scratch
packing buffers are dynamically allocated, not variable-length stack arrays.

Generated records retain inline field arrays sized for this module. Existing
C struct assignment and return therefore retain value-copy storage; I do not
replace records with aliased heap pointers. This uniform representation can
waste space when one record is much wider than others. Per-layout storage and
host stack consumption remain considerations, not claims resolved by this
change. Record arrays still have their existing 256-entry runtime capacity.

## Verification

`make -j1 test-nvm2c` passes 1,029 checks on Darwin. New executables exercise
75-, 129- and 300-field mixed integer/string records through both branch paths,
function arguments and returns, and local storage. They read high-index integer
and string fields. The former 75-field refusal test now compiles and executes;
its malformed branch-height variant still rejects. Existing negative field,
representation, array and namespace-collision tests remain passing.

I also compiled the test harness and `src/nanoisa/nvm2c.c` with
`-fsanitize=address -fno-omit-frame-pointer -O1 -g`. With `bin/nvm2c` supplied
as the harness's required CLI argument, this run passes the same 1,029 checks
without an AddressSanitizer report. The other linked objects and generated
executables were not sanitizer-instrumented; this is not a whole-runtime
sanitizer claim. My first invocation omitted the CLI argument and failed that
harness precondition. `git diff --check` passes.

`make -j1 test-one-ir-compiler` still fails, now with function 20 reporting
`AGG_PACK fields must be int or string`. Width is no longer its first failing
check. Array-valued fields, recursive aggregate shape information and full
compiler execution remain unfinished under MAC
`task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
