# Exact affine scalar-union runtime checkpoint

I record this bounded checkpoint for `task_a18a9f752536469faafc4d3ebec01dfd`.
It extends metadata checkpoint `d9ef8d969faf1dd456d167bd0534cffde6af97d1`;
it is not source-language admission, integrated platform acceptance or a release
claim.

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

## Focused results

On Darwin, from a clean source index with only this checkpoint's listed files
modified:

```text
make -j8 test-ownership-contracts
  PASS: 177 ownership-contract checks

make -j8 test-affine-bytecode
  PASS: 502 ordinary affine-bytecode checks
  PASS: 812 allocation-failure affine-bytecode checks

make -j8 test-verifier test-nvm2c test-owned-transfers
  PASS: 96 verifier checks
  PASS: 2422 structured nvm2c checks, 0 failed
  PASS: 184 ordinary and 275 allocation-failure owned-transfer checks
```

The combined adjacent command also invoked two Python sanitizer/link fixtures
with the default Apple tool selection. Those did not qualify: LeakSanitizer is
unsupported by that Apple runtime, and one fixture did not inherit Homebrew's
OpenSSL library path. I do not report those environment terminals as union
regressions or as passing evidence. The later source/runtime acceptance must use
the repository's explicit Homebrew LLVM and resolved linker selections.

## What remains

I still require an executable generated-C/VM union fixture, exact source
metadata emission for multiple concrete instances, statement and value match
lowering through both producers, and fresh integrated Linux and Darwin gates.
PR522 and release publication remain held.
