# NanoLang factual claim ledger

## Current authority

- `README.md` describes the public project boundary.
- `docs/PERSONA.md` defines my first-person voice.
- `docs/NANOISA.md` describes the current NanoISA and NanoVM architecture.
- `spec/nanoisa.yaml` is the active ISA schema source.
- `src/nanovirt/codegen.c` lowers NanoLang to NVM bytecode.
- `src/nanovm/vm.c` executes decoded instructions and handles traps.
- `src/nanoisa/verifier.c` validates current module and instruction invariants.
- `src/nanoisa/verifier_types.c` infers operand types across basic blocks.
- `src/nanoisa/nvm_format_v2.[ch]` and `src/nanoisa/nvm_v2_*.c` are the v2
  container and its section codecs.
- `src/nanovm/heap_cycles.c` is the cycle collector.
- `docs/NANOISA_MEASUREMENTS.md` is the performance authority.
- `tests/nanoisa/`, `tests/nanovm/`, and `tests/nanovirt/` are executable evidence.
- `docs/ROADMAP.md` records unresolved platform and example-catalog work; it is
  not release evidence until its named acceptance gate passes.

- `docs/NSI.md`, `docs/NSI_FABRIC.md`, `docs/NSI_EFFECTS.md`, `docs/NSI_TCB.md`,
  and `src/nsi*.c` are the NSI, fabric, policy, journal, and observability
  authorities.
- `docs/NANO_EMACS.md` and `modules/nano_eval/nano_emacs_worker.c` are the
  isolated editor walker.
- `docs/FORTH_2012.md` pins Forth 2012; Jackson suites are evidence, not a
  Standard System claim.
- `catalogs/messages/` and `userguide/i18n/` are catalogs and machine drafts.
  JSON/TOON stay English.

## Release evidence

The release before this edition is `v5.0.0`. This local 5.1 candidate edition
does not claim external document publication.
`docs/RELEASE_5.1.md` is the current candidate authority;
`docs/RELEASE_5.0.md` preserves the narrower prior cut.
The verified local counts at `v4.0.0` were 2,632 NanoISA tests, 621
NanoVM tests, 63 NanoVirt tests, and 93 verifier tests. 4.1–4.5 add Forth,
catalog, NSI, fabric, `nano_emacs_worker`, policy, journal, and
observability suites on top of that.

The NanoISA benchmark covers seven workloads plus cold startup and the
co-process boundary. Each workload is timed twice per sample -- once with a
single iteration and once with many, behind one process startup -- so the
per-iteration cost is the difference and the startup terms cancel. Before 4.0
the suite timed one process per sample and therefore measured startup rather
than execution; `docs/NANOISA_MEASUREMENTS.md` records both the corrected
baseline and that finding.

`docs/NANOISA_MEASUREMENTS.md` is the authority for every performance number I
present, including the optimizations I measured and declined.

## Diagnostics

`NANO_VM_TRACE` is read once in `vm_init` and guards per-instruction NanoVM
diagnostics. `NANO_PROFILE` is read once by generated-C profiling hooks and
guards timing collection. These are tested in `tests/test_dynamic_profile.sh`
and the NanoVM test suite.

## Boundaries

My historical 4.0 account covers the v2 module format, modeled verifier checks,
dispatch and measurements. The verifier tracks an abstract balance of explicit
retain/release instructions, not object identity or complete source ownership.
Unknown operand types are not proved safe. Current boundaries are recorded in
`CONTRIBUTING.md`, `docs/CANONICAL_STYLE.md` and the open roadmap.

What remains is roadmap work and is labelled as such. Module signing is
unimplemented; the 5.1 release number does not establish it.
LLVM and WebAssembly return only as NanoISA translators. I do not claim a
Forth Standard System, GNU Emacs, a kernel, or that the system is
internationalized. The trap journal is a tested library, not a hook on
every `vm.c` trap.
`Makefile.gnu` now generates and includes transitive C header dependencies.
The #211 roadmap item records its implementation and acceptance; the focused
gate is `make test-make-header-dependencies`. This does not close every cache
snapshot, publication or toolchain-identity requirement.
The mascot is an existing NanoLang user-guide asset, copied into this package
as `assets/nanolang-mascot.png`; external artwork is not authority for my claims.

My project requires useful shadows, while current CPU typechecking warns about
missing ones with documented exemptions. All three drivers select dependency
shadows by default; explicit root-only mode remains available. Separate test
processes have deadlines, not security sandboxing. `make test-language-claims`
checks the compiler observations; formal correspondence remains separate.
The shared nonnegative-input `examples/gcd.nano` demonstrates executable assertions,
not a theorem for every integer. Artifact tests extract and compile its text.

## 5.1 candidate checkpoint — 2026-09-24

I have not completed or tagged 5.1. PR #522 remains draft while full hosted
platforms, coverage and all sanitizer partitions are under qualification. The following September 20 record is historical.

At compiler-source pin `d56d15ff6`, the VM gate produces identical 530,308-byte
modules with SHA-256
`e6cc1c67ce6c2a8dc2d9ea0729f02c3b157e12a01299d220cb531e80839fd143`.
Both verify, preserve the exact host closure and execute the final compiler's
test product. The guard records zero NanoLang-generated C compilation during
VM generations. The complete gate passes in 776.915 seconds. Standalone-native
equality also passes on this pin: 530,296 bytes, SHA-256
`d678e0e1fe9e290592721b265f204b2917fc9369bf8db32d1c28835cc2c3f2ba`.
Both native generations verify and build as strict C11 without `libnanovm`;
the final compiler compiles a verified hello module that executes. I retain
this route in `docs/evidence/pr522-implementation-2026-09-23/native-fixedpoint-d56d15ff6/`.
Each route compares its own exact host-path closure.

The same source pin passes all 32 instrumented One IR methods, including both
complete compiler paths and the real std artifact. The original generic/signature
and selected-ownership suites pass 27 and 36 methods with instrumented generated
products, leak/UAR checks and unchanged deadlines. These bounded checks do not
replace complete hosted acceptance. The source pin and retained evidence are in
`docs/evidence/pr522-implementation-2026-09-23/vm-fixedpoint-d56d15ff6/` and
`docs/evidence/pr522-implementation-2026-09-23/final-source-d56d15ff6/`.

## 5.1 release-edition checkpoint — 2026-09-20

`docs/RELEASE_5.1.md` and `docs/NANOISA_ONLY.md` are my One IR authorities. At
compiler-source pin `ebe3afddc`, my NanoVM fixed-point gate produced identical
491,788-byte Stage 1 and Stage 2 modules with SHA-256
`7b8f96e51a146364448a37cce43dd414734d31aad53eb57359c6768a107f1575`.
Both verified, and Stage 2 compiled, verified and executed hello. The gate took
821.266 seconds. This proves raw reproducibility for that source and immutable
host closure, not compiler correctness.

At the same compiler-source pin, the standalone native route produced equal
491,800-byte Stage 1 and Stage 2 modules with SHA-256
`c3a425bb90edc4101a1e7c1ec92f2acee1a781a30407d608cfc381ea3a13c2b4`.
The two generations took 1,238.087 and 1,286.890 seconds, verified, preserved
the exact three-library host closure and compiled a verified, executing hello.
I do not compare those bytes with the distinct VM-route modules. The exact
record is `docs/evidence/v5.1-final-fixedpoints.md`.

My product driver publishes verified `.nvm`; `nvm2c`, `nvm2llvm` and
`nvm2wasm` consume the module. The C seed remains a bootstrap/reference
frontend. Private non-admitting mixed record-array execution, unrestricted
ownership profiles, isolated callbacks and production service isolation remain
outside the public claim.

## 5.0 release-edition checkpoint — 2026-09-16

I use `docs/RELEASE_5.0.md` and `docs/CALLBACK_ABI.md` for current contract
and callback claims. `src/runtime/callback_runtime.c`, `src/nanovm/vm_callback.c`,
the dispatch/SDL_mixer adapters and their tests implement retained handles,
owner-thread execution and cancellation. C-seed callback shadows select the
shared VM bridge. These in-process adapters do not establish isolated callbacks.

Dated finalization evidence records 1,739 native translator checks, 1,073 shape
checks, 272,403 VM checks, 89 VM codegen checks and 175 verified/equivalent
programs. A clean bootstrap passed at `4373abc5`. Darwin passed 14 effect
tests, 49 scoping checks and 39 executable guide snippets; ownership sanitizers
exercised 52,000 activations. Strict Linux ARM64 acceptance passed all 185
selected native example artifacts with unchanged selection/exclusions and five
root/examples-working-directory regression compilations. These are bounded checkpoints. Exact-commit
clean-tree tests, platform CI and release acceptance are mandatory release gates. `docs/evidence/main-reconciliation-pr334-linux.md` owns the detailed
integration evidence. Full NanoISA-only bootstrap and backend parity remain open.
