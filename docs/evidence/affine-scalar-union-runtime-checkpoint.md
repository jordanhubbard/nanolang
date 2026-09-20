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

My ownership verifier admits the bounded union opcodes only after the shared
version-3 envelope and its `UNION_VARIANTS` extension validate. The envelope
bounds the unchanged version-2 path encoding with `path_bytes`, then carries
ordered mandatory-understanding extensions. Kind 1 is `UNION_VARIANTS` and
kind 2 is reserved for the independently owned `ARRAY_FIELDS` validator. The
current common reader refuses kind 2 rather than projecting only the union
facts. A scalar-union constructor counts as the refcounted runtime
transfer it creates. I keep version 3 scalar-union modules out of the unrelated
owner-ARRAY route. Generated native C retains and releases union carriers,
records layout, variant and payload count at construction, validates returned
union boundaries, and lowers `MATCH_TAG`, checked projection and terminal
failure without inventing a default value.

The focused fixtures cover heterogeneous and empty variants, two simultaneous
concrete generic spellings, exact constructor and projection refusals, unknown
parameter refinement, known taken and untaken branches, public verifier
admission and native source emission.

I no longer use those display spellings as the equality authority. The
self-hosted producer derives a key from the resolved root-module union
declaration, its module owner and every recursively resolved type argument.
Canonical spelling differences therefore share one key, while simultaneous
and nested concrete instances remain distinct. The admitted producer still
refuses imports. An unresolved alias, ambiguous declaration or equal-looking
cross-module declaration therefore refuses instead of falling back to text;
this checkpoint does not claim imported-alias admission.

`MATCH_TAG` now refines only the stack value it tests on the successful edge.
It does not write the selected variant back to the source local. A later load
of that local, another parameter with the same concrete layout and a join with
an unrefined predecessor all require their own proof before projection. Direct
projection from the matched stack value remains admitted.

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
  PASS: 185 ownership-contract checks, including bounded v2 paths inside v3,
        exact extension framing, zero padding, ordering, uniqueness,
        mandatory-understanding revisions/kinds and held ARRAY_FIELDS refusal

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
  PASS: producer bytes contain the bounded v2 path suffix and framed
        UNION_VARIANTS kind 1/revision 1 payload
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

## Declaration identity and exact-value qualification

I qualified the bounded identity and refinement correction from a fresh
detached Darwin checkout at exact head
`19c7573ed9645f89f058012a782fe4a3833122a8`. That head is the bounded
production checkpoint `143ad7ab` merged with canonical main `af8809b3`; the
merge was automatic and retained the shared version-3 envelope unchanged.

The first new different-receiver control did not reach affine analysis. The C
fixture emitted one `.parameters` tag for every nonzero arity, so its new
two-parameter function was malformed. I retained that terminal and corrected
the fixture to emit the exact requested parameter list before rerunning any
acceptance gate.

```text
make -j8 bootstrap
  PASS in 301.86s: C seed, Stage 1, Stage 2 and installed-compiler smoke checks
  NOTE: Stage 1 and Stage 2 native binaries differ; this is not fixed-point evidence

make -j8 nanoisa_emit nano_vm nvm2c nanoisa_dump
  PASS in 76.89s

make -j8 test-affine-scalar-union-runtime
  PASS: 546 ordinary affine checks
  PASS: 856 allocation-path affine checks
  PASS: VM/native runtime and Homebrew LLVM 23 ASan/UBSan/LSan

NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang \
  python3 -m unittest -v tests.test_affine_scalar_union_source
  PASS: simultaneous concrete instances verify and execute in VM/native routes

make -j8 test-ownership-contracts test-owned-transfers
  PASS: ownership contract Python control
  PASS: 184 ordinary and 275 allocation-path owned-transfer checks

make -j8 test-verifier test-nvm2c
  PASS: 96 verifier checks
  PASS: 1,365 shape constraints
  PASS: 2,422 structured nvm2c checks, 0 failed
```

The source index was clean before qualification and after the final command.
The selected actual compilers
were Apple Clang 21.0.0 at SHA-256
`1590ac950a3d627817d09ade5cb60b2115f17a72182a3141e010b4bcc482a0c9`
and Homebrew Clang 23.1.1 at SHA-256
`570c488e53383b198796e706e91b5ce5ec45bb730683a5af5e822d56a2eb1888`.
The retained raw logs are:

```text
e7feda8dd8688be8f914e6c6e4f3ad04c6329403bceed9617bd1b578a0148b03  /private/tmp/nanolang-pr893-19c7573e-bootstrap.log
290aca9b9257535383d31e8eeb866945ee3cabb080bf1bb2cf3cce98aa218a8d  /private/tmp/nanolang-pr893-19c7573e-tools.log
740d56b846a83ef4c576a2d94d460ad6ae0fc2653886fb374949bdbcc88ac552  /private/tmp/nanolang-pr893-19c7573e-runtime.log
94c3cf1d571110a2ece5a2ac87a2c64f440e3eed95945fb79987d8a57b307036  /private/tmp/nanolang-pr893-19c7573e-source.log
0590ffbe33cfebe0f4c8b0432e5ddf8f0bdd53a5c4a74e181555bc6d78989bf9  /private/tmp/nanolang-pr893-19c7573e-adjacent.log
900e234dd550791d95330664079a192c7bdf6bfa155fb51f7ae42852901bd3fc  /private/tmp/nanolang-pr893-19c7573e-verifier-nvm2c.log
```

## What remains

I still require the complete statement/value match matrix, precise
unsupported-payload refusals, qualified mixed-envelope conjunction after the
independently owned `ARRAY_FIELDS` validator, Stage 1/Stage 2 producer parity
and fresh integrated Linux qualification. PR522 and release publication remain
held.
