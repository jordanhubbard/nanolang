# NanoLang 5.1.0 candidate — One verified compiler product

I am preparing 5.1.0; this is a candidate record, not a completed release.
Final platform, sanitizer and fixed-point qualification remain open in
[PR #522](https://github.com/jordanhubbard/nanolang/pull/522).

I publish verified NanoISA bytecode as my portable compiler product.
Unqualified compilation writes a sibling `.nvm`; native executables and C11
source are produced from that verified module through `nvm2c`. My self-hosted
compiler no longer needs its NanoLang-to-C pretty-printer in the product
dependency closure.

This is the architecture release that my narrower 5.0.0 cut deliberately left
open. I preserve that earlier tag and its boundaries in
[my 5.0 release record](RELEASE_5.0.md).

## One IR

My supported product path is now:

```text
.nano source → checked AST → verified .nvm v2
                                  ├─ nano_vm
                                  ├─ nvm2c → C11 → native process
                                  ├─ nvm2llvm
                                  └─ nvm2wasm
```

- A source path with no explicit output publishes a sibling `.nvm`.
- `-o <binary>` and `--target native` translate the verified module through
  `nvm2c` and the selected C compiler.
- `--target c` publishes the C11 translation of the verified module.
- VM-generation shadows execute as bytecode. Dependency shadows run before
  root shadows unless `--root-shadows-only` is selected.
- Module facts, diagnostics, guest arguments, source locations, artifact
  imports and declared host signatures survive the product boundary.

The C implementation remains my bootstrap seed and a reference frontend. It
is not a second product IR. I retain older source files and explicit research
paths where they still provide bootstrap or compatibility evidence; their
presence does not make them the default compiler product.

## Self-hosting evidence

My latest recorded VM fixed point is at `0eb94026c`: both generations contain
526,188 bytes with SHA-256
`19e49b0210f46e1bd7490ff0efa0e49f24bf42d924c3bbafb26af42850ef474b`.
Both verify, retain the same host closure, and the final compiler produces a
verified executable test module. The gate records zero NanoLang-generated C
compilation calls during VM generations. My [retained record](evidence/pr522-implementation-2026-09-23/vm-fixedpoint-0eb94026/README.md)
preserves the exact pin and limits. The subsequent native translator correction
still needs final-source VM and native qualification.

My standalone native fixed point at `e70c0de46` also passes: successive
526,188-byte modules have SHA-256
`2c107f991005a04259a9978e2609ad97be5c45299712f9ca0c223d63e4d70ca5`.
Both verify with a stable host closure. I translate and build the final
compared module, then compile, verify and execute hello with it. My
[retained native record](evidence/pr522-implementation-2026-09-23/native-fixedpoint-e70c0de46/README.md)
contains the complete gate and hashes. This pin predates subsequent C-seed
metadata cleanup; it does not qualify later source revisions.

The following earlier results remain historical evidence.

At compiler-source pin `ebe3afddc8c9a7cd89b5d64b2928ea9dd269d08c`, my
NanoVM bootstrap produced two successive 491,788-byte compiler modules with
the same raw SHA-256:

```text
7b8f96e51a146364448a37cce43dd414734d31aad53eb57359c6768a107f1575
```

Generation 1 took 380.815 seconds and generation 2 took 385.655 seconds on the
recorded Linux host. Both modules verified. The second generation compiled,
verified and executed the unchanged hello program, and the declared host
closure remained unchanged. The complete gate took 821.266 seconds. I compare
raw Stage 1 and Stage 2 bytes; I do not normalize them or compare the distinct
C-seed lowering output with Stage 1.

The standalone native route at the same source pin produced two successive
491,800-byte modules with raw SHA-256
`c3a425bb90edc4101a1e7c1ec92f2acee1a781a30407d608cfc381ea3a13c2b4`.
Generation took 1,238.087 seconds and 1,286.890 seconds. Both modules verified;
the Stage 1 native compiler compiled the unchanged hello source, and that
module verified and executed. The exact three-library host closure and
post-run source/tool hashes remained unchanged. The native compiler processes
do not link `nano_vm`.

The VM and native artifacts have different sizes, so I make no cross-route
raw-equality claim. [My retained fixed-point record](evidence/v5.1-final-fixedpoints.md)
names the bounds, hashes and evidence limits. A fixed point is reproducibility
evidence, not a proof that my compiler is correct.

## Language and runtime work carried into 5.1

- I preserve lexical first-success match ordering. `return` inside a match or
  effect-handler arm exits the enclosing function; the arm's final expression
  supplies its value.
- I carry concrete generic-union identity and exact scalar payload layouts
  through imports, matches, verified metadata, NanoVM and native AOT.
- I lower records, tuples, maps, scalar and nested arrays, strings, optional
  values, floats and bytes through the canonical emitter. Computed integer to
  `u8` conversion is checked at destinations and before tail-return selection.
- I retain ownership and borrow facts across calls, results, callbacks,
  globals, loops, snapshots and cleanup. Unsupported or incomplete profiles
  fail before publication rather than silently changing representation.
- I retain lifetime-safe in-process callbacks with signatures, owner-thread
  execution, cancellation and cleanup. Callback-bearing isolated imports
  remain unsupported.
- I provide NanoISA translators for C11, LLVM IR and WebAssembly. Restricted
  GPU and direct experimental paths remain explicitly profiled rather than
  being described as general backends.
- I carry checked File service plans through the qualified public VM/native
  route with explicit host grants. Private experimental execution does not
  widen public authority.

## Verification and evidence

At `e70c0de46`, both complete native translator gates pass 2,524 assertions,
including a fresh ASan/UBSan build; the shape solver passes 1,565 assertions.
All nine original functional-array methods pass with generated ASan/UBSan and
leak detection. Fresh Darwin bootstrap passes both native stages and installed
execution without the C seed. Strict GCC builds the corrected full compiler
output on Linux, and that standalone compiler produces a hello module that
verifies and executes. My [retained correction evidence](evidence/pr522-implementation-2026-09-23/array-consumer-inference/README.md)
includes the failures and qualification boundaries.

These are local checks at a recorded checkpoint. They do not replace the
complete clean-tree suite, every hosted platform and sanitizer partition, or
final-source fixed points required before the tag.

My formal NanoCore proofs remain `Admitted`-free for their stated model. They
do not prove the full compiler, foreign code, service adapters or every
runtime representation.

## Deliberate boundaries

- PR #936 is a private, non-admitting mixed record-array VM experiment. It is
  not part of the public 5.1 product and grants no source or service authority.
- My checked ownership profiles are substantial but deliberately refuse
  shapes whose lifetime or alias contract is not established. I do not claim
  that all possible resource graphs are accepted.
- `nano_cop` isolates supported VM foreign calls. Native AOT uses its declared
  host ABI; it does not pretend that an in-process C call is isolated.
- My NSI, capability fabric and journal remain runtime foundations on an
  ordinary kernel. I do not claim a kernel, a Forth Standard System, GNU Emacs
  compatibility, reviewed human translations, or universal backend parity.

## Release presentation

I regenerate the local developer deck and narrative for 5.1 before tagging.
Publishing those files to the existing Google Slides and Docs resources is a
separate, explicitly initiated action; the release does not silently replace
external documents.
