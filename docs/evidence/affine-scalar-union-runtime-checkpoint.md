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

## Fresh Linux qualification

I used a separate shared-object clone on sparky at
`/tmp/nanolang-pr893-bb5d744b.mYM3aX`; I did not mutate the active root
checkout. The first setup command stopped before checkout or build because the
clone inherited the local checkout as its origin and that checkout did not
carry the pushed PR ref:

```text
fatal: couldn't find remote ref fix/affine-scalar-unions-pr889
```

I retained that terminal in this report, changed only the isolated clone's
origin to the canonical GitHub repository and checked out exact head
`bb5d744beba6086c82ee2f0778802b86b4a1f75c`. Fresh bootstrap passed in
274.96 seconds and the required tools built in 27.33 seconds. Stage 1 and Stage
2 again passed their smoke checks but produced different native binaries; this
is not fixed-point evidence.

The first affine runtime command then stopped before executing its checks.
GCC 13 rejected the fixture's one-line failed-assembly diagnostic followed by
`CHECK` under `-Werror=misleading-indentation`. Its retained log is
`2622ab6fc99e49798c9c6e7acbc7fc152a4ca3852ed7d96c20cfd0fe5e2914ab`.
I braced only that diagnostic branch at `c30543c3c`; affine production and the
qualified compiler binaries remained byte-identical. The corrected fixture
also passed the 546/856 affine and VM/native sanitizer runtime gate on Darwin;
its raw log SHA-256 is
`ab982cf48888f6229ee0668b7c87eaf35fde453765fa3e9a2ba07b348a03abe4`.
From that test-only head on Linux:

```text
make -j8 test-affine-scalar-union-runtime
  PASS: 546 ordinary and 856 allocation-path affine checks
  PASS: VM/native runtime under GCC 13 ASan/UBSan with leak detection

NANO_NATIVE_TEST_CC=cc python3 -m unittest -v tests.test_affine_scalar_union_source
  PASS: simultaneous concrete instances verify and execute in VM/native routes

make -j8 test-ownership-contracts test-owned-transfers
  PASS: ownership contract Python control
  PASS: 184 ordinary and 275 allocation-path owned-transfer checks

make -j8 test-verifier test-nvm2c
  PASS: 96 verifier checks
  PASS: 1,365 shape constraints
  PASS: 2,422 structured nvm2c checks, 0 failed
```

The source index was clean after the final command. The actual compiler was
Ubuntu GCC 13.3.0 at `/usr/bin/aarch64-linux-gnu-gcc-13`, SHA-256
`a20520ee21543f243d40636a9181a142c45ecd989de31ab86b99a8ea5ada870d`.
The corrected retained logs are:

```text
7b10411b947aa87596a6d6d0a134c1ee295afc4f8cc44cbb89779965ee54d594  /tmp/nanolang-pr893-bb5d744b.mYM3aX/bootstrap.log
d49a6b4b1dc1f35029923ca283557ee013b61d70bed9d9c42caaaa42e45d663e  /tmp/nanolang-pr893-bb5d744b.mYM3aX/tools.log
0a93f8cd8d91e7455cb851e2bce29019fac9ca90e5f6515d6f29d96379105653  /tmp/nanolang-pr893-bb5d744b.mYM3aX/runtime-corrected.log
3ad904e88033d857cbea88985aab392e2f8093c8ffe60bcf9c1f8b65256b18cb  /tmp/nanolang-pr893-bb5d744b.mYM3aX/source-corrected.log
491f53dd7dd29e85c44682ac4f58c9a8bd261604e11acfb75a533581afd702bb  /tmp/nanolang-pr893-bb5d744b.mYM3aX/adjacent-corrected.log
3353008c44f756c56a2960d26a5992b31238ce191a970db1d41ec97b0e2e0d46  /tmp/nanolang-pr893-bb5d744b.mYM3aX/verifier-nvm2c-corrected.log
```

## Current-main integration after PR910

Canonical main advanced to `ca0c9f6a8` while the preceding gates ran. That
history includes the generic-return match repair in merged PR910 and the
portable read-text Wasm work. I merged it without conflicts at
`84625401cdaf0bff52ebef92360d2243a16e0a66`. The merge changed none of this
checkpoint's producer, affine-analysis or focused-test files; their SHA-256
values remained:

```text
156547084b021e77b98667289a01a3e8a9a39187e7ec4f3689260d743ab85089  src_nano/compiler/nanoisa_borrows.nano
1a2b381e40022a3ca2b7f9920c17fbf650cc11b983927ddbb5356151add6837a  src/nanoisa/affine_bytecode.c
0f16fbab71e2b5f6d89368cdc079582dd709c98abea079e96db3f9dd43ca3bb6  tests/nanoisa/test_affine_bytecode.c
```

I nevertheless rebuilt and reran the complete bounded qualification at that
exact integrated head on both platforms. Fresh Darwin bootstrap passed in
330.45 seconds and the tools built in 90.09 seconds. Fresh Linux bootstrap
passed in 267.40 seconds and the tools built in 27.44 seconds. Both platforms
then passed 546 ordinary plus 856 allocation-path affine checks, VM/native
sanitizer execution, the simultaneous-instance source route, ownership
contracts, 184 ordinary plus 275 allocation-path transfers, 96 verifier checks,
1,365 shape constraints and all 2,422 structured nvm2c checks. Both source
indexes were clean afterward. The current-head logs are:

```text
9782e2747598e93056fc67b34a26e6e9c77323378aa0b6139f6d9a802a4e4c9e  /private/tmp/nanolang-pr893-84625401-darwin-bootstrap.log
f0ac9e395af2fbd169c0864d4cca1e8eea36cd9a93cb885d8dba2906114142ee  /private/tmp/nanolang-pr893-84625401-darwin-tools.log
4538cf23d4677ac1364b206b3ee3b2c1546e660cbf77f7c7a22239a502359759  /private/tmp/nanolang-pr893-84625401-darwin-runtime.log
9f975901e3b9e1e5d76564e4aa6ffd139a1ffa9e805438c9dec892ca40b8dabd  /private/tmp/nanolang-pr893-84625401-darwin-source.log
460364d342f9e2b59a818ad2b3c38b6582821b247eb1e3319a0999b3c9774121  /private/tmp/nanolang-pr893-84625401-darwin-adjacent.log
e602705c75429db096e4faffe8afdc16ff83cf5bf962c65eabab1735a24d4695  /private/tmp/nanolang-pr893-84625401-darwin-verifier-nvm2c.log

b38111b88dda9784183893f299f48e2b41015cf0e18f34f2b06591404f9f2106  /tmp/nanolang-pr893-bb5d744b.mYM3aX/current-bootstrap.log
e9d4e73b0bb45703159ce8b09e363378aadbbd4e7636077b2498f5e1bf9b9088  /tmp/nanolang-pr893-bb5d744b.mYM3aX/current-tools.log
b8e65f9521bbed707aa441cb4cf17f6138882c9bf10f6b7ee2e35382072e8e13  /tmp/nanolang-pr893-bb5d744b.mYM3aX/current-runtime.log
8fcc7f0934c276f56420fb10b9fbf85066c3e3fcbe6169015826f635b3e7e7b0  /tmp/nanolang-pr893-bb5d744b.mYM3aX/current-source.log
d59170b338f8f11b4234108600263331c8fed44faf533fc9ceca1a729502aadd  /tmp/nanolang-pr893-bb5d744b.mYM3aX/current-adjacent.log
ab7c086173af223b3ec48076c5ad6d6f9777495bbb59e4e3b3b698e080fd6653  /tmp/nanolang-pr893-bb5d744b.mYM3aX/current-verifier-nvm2c.log
```

## Additive current-main restack

Canonical main then advanced to `0061feed5` through private cyclic File dispatch
and exclusive File binding publication. Those commits changed `Makefile.gnu` and
my roadmap additively, but changed none of my union producer, affine analysis,
ownership codec, verifier, VM, nvm2c or focused fixtures. I merged that exact
main automatically at `dac87fbecbea5285d21ca376a42fbb9e9283b2b3`; the three
scoped source hashes remained byte-identical to the preceding qualification.

I did not call this unrelated File integration a new compiler bootstrap. I ran
the complete bounded union targets at the exact merged head with freshly linked
test products. They pass 546 ordinary plus 856 allocation-path affine checks,
VM/native sanitizer execution, the simultaneous-instance source route,
ownership contracts, 184 ordinary plus 275 allocation-path transfers, all 96
verifier checks, 1,365 shape constraints and all 2,422 structured nvm2c checks.
The retained Darwin logs are:

```text
0021042f3e9cc43f8d23f46b7564573c8bd8f9c0740ddfe9f2b5c363246a3bb1  /private/tmp/nanolang-pr893-dac87fbe-darwin-runtime.log
7415ebc66b7a29a323e90fae5aa514a2747c9b0598fe30072d6fc632015a1471  /private/tmp/nanolang-pr893-dac87fbe-darwin-source.log
9c7c531f85367e2d6630eaaab13f85ae07c4f1bcf5469149e52df95edf3515ff  /private/tmp/nanolang-pr893-dac87fbe-darwin-adjacent.log
05e8586680dacfa3200d70beb657dd555dcb70ad434a2f6eefa5cb19c754e4b8  /private/tmp/nanolang-pr893-dac87fbe-darwin-verifier-nvm2c.log
```

## What remains

I still require the complete statement/value match matrix, precise
unsupported-payload refusals, qualified mixed-envelope conjunction after the
independently owned `ARRAY_FIELDS` validator, Stage 1/Stage 2 producer parity
and the remaining full-product qualification. PR522 and release publication
remain held.
