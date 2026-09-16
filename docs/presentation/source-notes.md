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

My local repository contains `v4.0.0` and `v4.5.0` tags. This local 5.0 draft
does not claim a new tag or external publication. `docs/RELEASE_4.5.md` is the
public summary covering 4.1–4.5. `docs/RELEASE_4.4.md` is the 4.4 product
page as it stood on `main` before Phase 19.
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
unimplemented; the 5.0 release number does not establish it.
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

## 5.0 candidate checkpoint — 2026-09-16

I use `docs/RELEASE_5.0.md` and `docs/CALLBACK_ABI.md` for current contract
and callback claims. `src/runtime/callback_runtime.c`, `src/nanovm/vm_callback.c`,
the dispatch/SDL_mixer adapters and their tests implement retained handles,
owner-thread execution and cancellation. C-seed callback shadows select the
shared VM bridge. These in-process adapters do not establish isolated callbacks.

Dated Linux finalization evidence records 1,739 native translator checks,
1,073 shape checks, 272,379 VM checks and 242 example programs. These counts
are checkpoints, not final-tree validation. Native/VM effect repairs are still
in progress. `docs/evidence/main-reconciliation-pr334-linux.md` owns the detailed
integration evidence. Full NanoISA-only bootstrap and backend parity remain open.
