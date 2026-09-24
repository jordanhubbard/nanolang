# Uncalled array parameter storage

I continue PR #522 from `3be3f485a886df54b396b974f49491ad58f14400` under
`task_f78020ad5c124c36bc26175241bc2539`.

My original functional-array gate fails native translation of
`tests/nl_functions_array_param.nano` with `STORE_LOCAL: expected string value`.
The preceding compiler-metadata-lifetimes evidence retains that terminal.
I retain the emitted module and its disassembly here. The module keeps every
helper body, while `main` only prints its status message. My separate mandatory
shadow module supplies the actual calls.

My representation inference used caller facts and scalar parameter declarations,
but left an uncalled declared array unresolved. Its element projection could
therefore fall through to integer local storage before a later string store.
After caller facts converge, I now represent a still-unresolved declared array
parameter with the existing tagged handle. This preserves the missing element
fact and propagates tagged reads through local copies; checked scalar consumers
retain their runtime tag requirements. Known caller facts retain their original
representations. I do not remove uncalled bodies or alter shadow selection.

I add eight raw-module controls: integer, boolean, float and string results,
each with both declaration orders. Each retained helper reads its parameter,
stores the projection in an intermediate local and copies it into an initialized
scalar local. Strict generated-C compilation checks all bodies; the independent
entry remains executable. The original source gate separately executes the
helpers in mandatory shadows and checks VM/native parity.

The complete ordinary translator gate passes 2,516 assertions; its prerequisite
shape gate passes 1,500 assertions. The unchanged functional-array gate passes
all nine methods with generated ASan/UBSan, no recovery and leak detection,
using Homebrew Clang and CI's existing 60-second shadow deadline. I retain the
strict GCC command's initial missing-include failure separately.

My complete fresh translator sanitizer gate also passes all 2,516 assertions
and verifies ASan/UBSan symbols in both translator and shape objects. It retains
the owning harness's existing leak-detection exclusion; the functional-array
gate above separately enables leak detection. Strict GCC 13 compilation of the
final test source passes with `-O3 -Wall -Wextra -Werror`. The ordinary run
preceded a buffer-size-only fixture adjustment; the sanitizer and GCC checks
cover the final buffer declaration.
Full hosted acceptance, canonical bytecode fixed points and release qualification
remain separate requirements.

My MAC task rejects direct completion from its open lifecycle state. I retain
the rejection; the repository qualification does not claim a closed ledger task.
