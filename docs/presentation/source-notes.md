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

## Release evidence

Release `v4.0.0` is tagged at the repository history. The release summary is
`docs/RELEASE_4.0.md`. The verified local counts at that tag are 2,632 NanoISA
tests, 621 NanoVM tests, 63 NanoVirt tests, and 93 verifier tests.

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

Phase 12 is complete: 78 of 78 items. The NanoISA v2 verifier, module format,
dispatch, ownership and measurement work is shipped, not roadmap.

What remains is later work and is labelled as such. Module signing is 5.0.
LLVM and WebAssembly return only as NanoISA translators, which is Phase 14.
`Makefile.gnu` tracks no header dependencies, so a struct change makes an
incremental build untrustworthy; that is filed as issue #211 and is a known
defect rather than a claim.
The mascot is an existing NanoLang user-guide asset, copied into this package
as `assets/nanolang-mascot.png`; external artwork is not authority for my claims.
