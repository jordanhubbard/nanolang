# Exact affine scalar-union runtime checkpoint

I record this bounded checkpoint for `task_a18a9f752536469faafc4d3ebec01dfd`.
It extends metadata checkpoint `d9ef8d969faf1dd456d167bd0534cffde6af97d1`;
it is not complete source-language admission, integrated platform acceptance or
a release claim.

## What I checked

I now keep a concrete union layout and a path-local selected variant in affine
state. My constructor checks the exact variant arity and substituted payload
tags. `MATCH_TAG` refines only its taken edge, and a payload projection requires
that proof. Known constructed variants prune their impossible edge rather than
rejecting an unreachable source-order arm. Locals, calls, results and joins keep
the concrete layout and conservatively forget only variant refinement.

My ownership verifier admits the bounded union opcodes only after version 3
metadata validates. A scalar-union constructor counts as the refcounted runtime
transfer it creates. I keep version 3 scalar-union modules out of the unrelated
owner-ARRAY route. Generated native C retains and releases union carriers,
records layout, variant and payload count at construction, validates returned
union boundaries, and lowers `MATCH_TAG`, checked projection and terminal
failure without inventing a default value.

The focused fixtures cover heterogeneous and empty variants, two simultaneous
concrete generic spellings, exact constructor and projection refusals, unknown
parameter refinement, known taken and untaken branches, public verifier
admission and native source emission.

My self-hosted affine producer now registers canonical concrete spellings
independently of source union declarations. It emits one retained layout per
instance, exact variant-major payload fields, version-3 variant slices and
concrete union descriptors for parameters, locals and results. The first source
fixture carries `Choice<int,string>` and `Choice<float,bool>` through separate
functions in the same module. Their `AGG_PACK` ordinals are 0 and 1; neither
silently selects the other's layout.

That fixture exposed a route-order defect: every version-3 module containing a
resource record was rejected as an owner-ARRAY candidate before its fields were
read. I now select that route by inspected resource-array fields. Ordinary
resource records plus scalar unions continue through the affine verifier. A
version-3 module that actually combines owner arrays and unions remains refused.

## Focused results

On Darwin, from a clean source index with only this checkpoint's listed files
modified:

```text
make -j8 test-ownership-contracts
  PASS: 177 ownership-contract checks

make -j8 test-affine-bytecode
  PASS: 503 ordinary affine-bytecode checks
  PASS: 813 allocation-failure affine-bytecode checks

make -j8 test-verifier test-nvm2c test-owned-transfers
  PASS: 96 verifier checks
  PASS: 2422 structured nvm2c checks, 0 failed
  PASS: 184 ordinary and 275 allocation-failure owned-transfer checks

make -j8 test-affine-scalar-union-runtime
  PASS: serialized artifact verifies and executes in NanoVM
  PASS: command-line nvm2c reproduces the in-process generated C byte-for-byte
  PASS: strict Homebrew LLVM 23 ASan/UBSan/LSan native execution
  PASS: every injected allocation failure releases all roots; successful result is true

make -j8 test-affine-scalar-union-source
  PASS: one owner-transfer graph emits two distinct concrete union layouts
  PASS: the assembled artifact verifies and executes in NanoVM
  PASS: the dump retains both concrete spellings
  PASS: generated C compiles with -Wall -Wextra -Werror and Homebrew LLVM 23 ASan/UBSan/LSan
```

The combined adjacent command also invoked two Python sanitizer/link fixtures
with the default Apple tool selection. Those did not qualify: LeakSanitizer is
unsupported by that Apple runtime, and one fixture did not inherit Homebrew's
OpenSSL library path. I do not report those environment terminals as union
regressions or as passing evidence. The later source/runtime acceptance must use
the repository's explicit Homebrew LLVM and resolved linker selections.
An additional owner-ARRAY authority invocation first lacked Homebrew's OpenSSL
library path; with that path supplied, its existing public-boundary executable
terminated with status -11 after reporting an unresolved parameter boundary.
I retain that terminal as unqualified adjacent evidence and do not attribute it
to scalar unions.

## What remains

I still require statement and value match lowering through the affine producer,
precise unsupported-payload refusals, Stage 1/Stage 2 parity and fresh integrated
Linux and Darwin gates. PR522 and release publication remain held.
