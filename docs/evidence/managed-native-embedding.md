# I qualify exact standalone managed C embedding

I retain task24c910 and fixture repaira522 under mixed runtime78c9. My embedding
production9c8194b8 is unchanged by these checks;207ebfe3 narrows only its generator
comment to supported ASCII-compatible C toolchains. I grant no mixed bytecode,
source or public backend admission from this qualification.

My first frozen207ebfe3 gate passed exact byte/hash equality, isolated regeneration
and refusal controls, and GCC O0/O2 compilation/execution. Strict Clang O0/O2 refused
three unused arithmetic helpers because my new fixture exercised only add. I
preserve that terminal status1 and its1.120seconds. No refused executable ran.
I recorded fixture childa522 before adding ordinary sub/mul/div assertions.

Correctedfe321533d13fbf016fcff7af2d9030e35f89c250 passes all three methods in
1.622seconds. Four standalone C programs compile with `-std=c11 -Wall -Wextra
-Werror -pedantic` at O0/O2 using GCC13 and Clang23. Clang uses my retained wrapper
with `--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`; I retain actual compiler,
Python, assembler/linker, GCC helper and wrapper identities before/after. Compile
commands use no repository include path, external runtime source or explicit link
library. ELF inspection records libc and no libm dependency for these programs.
This measures these host/tool configurations, not every platform or optimizer.

I compare the generated header's decoded bytes with the exact three source inputs
minus only the two named local includes. Their individual and assembled hashes
match. Isolated temporary copies test missing/stale output, deterministic repeated
regeneration, exact-once include removal and unknown-local-include refusal with
prior output retained. The existing parser embedding precedes the full runtime;
its guard prevents duplicate definitions. Existing arithmetic provider definitions
coexist and all four operations are exercised.

The ordinary runtime fixture creates a FLOAT array, stores it in an exact bound
ordinary descriptor, projects an alias and retires the parent. It checks the alias
value and missing-index VOID, releases it, requires zero live objects/bytes before
finish/dispose, and checks successful disposal. This is a small storage/lifecycle
control, not full managed allocation-fault qualification or VM/native mixed
execution. No sanitizer result is claimed here.

Root subsequently identified unconditional readelf as a Darwin discovery gap.
After its ledger/roadmap record,fcfbd803 changes dependency inspection to otool -L
on Darwin while retaining readelf on Linux. Compiled C and lifecycle assertions
are unchanged. I do not rerun identical Linux compilation for this platform-only
change, and I do not claim the Darwin branch has run.

My [manifest](managed-native-embedding/manifest.json) seals both terminals, exact
commands, logs, standalone C, per-executable hashes and unchanged before/after
source/tool inventories. Successful binaries remain in their `/tmp/nanolang-
managed-native-embedding-{207ebfe3,fe321533}/programs` directories; the repository
retains hashes rather than executable copies. No existing compiled library
objects are linked by these standalone commands. This evidence does not qualify
uncompiled private mixed preparation8952886d or close any runtime/source parent.
