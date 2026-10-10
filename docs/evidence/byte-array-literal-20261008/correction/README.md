# Contextual byte-array storage correction

I retain whether a literal's element kind came from a checked annotation.
Repeated expression checks validate its elements without erasing that storage
kind. I apply checked contextual kinds to nonempty argument, return and record
field literals as well as bindings and assignments. NanoVirt widens a byte at
an integer destination before its return-tag check; scalar byte literal range
policy is unchanged.

My initial paired test has eight failing subcases: NanoVM local binding and
assignment, plus C native and NanoVM return, argument and field contexts.
Both global cases pass. The first annotation correction leaves one failure:
the now-correct byte argument reaches an integer return without widening.
I retain both failures, then correct numeric destination conversion.

`make -j2 test-byte-array-literals` passes both methods in 8.912 seconds:
12 context/backend execution cases and 36 wrong-kind refusal cases preserving
prior output. `make -j2 test-typechecker test-nanovirt` passes the complete
typechecker suite and all 90 NanoVirt checks. The final Make gate builds and
uses this isolated worktree's own C seed, NanoVirt and NanoVM. Earlier focused
runs select the copied VM explicitly; they are not installed-product gates.
My source hashes record the implementation tested over base `1280314de`.

This is a C-frontend checkpoint under #979. Self-hosted byte/nested arrays,
native byte/nested slice translation and full fresh compiler qualification
remain required. I keep the original slice tests and legacy-retirement gate.
