# NanoLang factual claim ledger

## Current authority

- `README.md` describes the public project boundary.
- `docs/PERSONA.md` defines my first-person voice.
- `docs/NANOISA.md` describes the current NanoISA and NanoVM architecture.
- `spec/nanoisa.yaml` is the active ISA schema source.
- `src/nanovirt/codegen.c` lowers NanoLang to NVM bytecode.
- `src/nanovm/vm.c` executes decoded instructions and handles traps.
- `src/nanoisa/verifier.c` validates current module and instruction invariants.
- `tests/nanoisa/`, `tests/nanovm/`, and `tests/nanovirt/` are executable evidence.

## Release evidence

Release `v3.5.0` is tagged at the repository history. The release summary is
`docs/RELEASE_3.5.md`. The verified local counts recorded for the release were
1,077 NanoISA tests, 322 NanoVM tests, and 63 NanoVirt tests; subsequent 4.0
container and verifier work raised the NanoISA count to 1,090.

The NanoISA benchmark covers seven workloads with 20 samples per workload.
The recorded profile fields include retired instructions, opcode sequences,
branches, calls, stack and frame depth, traps, heap traffic, and FFI traffic.

## Diagnostics

`NANO_VM_TRACE` is read once in `vm_init` and guards per-instruction NanoVM
diagnostics. `NANO_PROFILE` is read once by generated-C profiling hooks and
guards timing collection. These are tested in `tests/test_dynamic_profile.sh`
and the NanoVM test suite.

## Boundaries

The remaining NanoISA v2 verifier, module, dispatch, runtime-representation,
FFI, and ownership work is 4.0 roadmap work. I do not present it as shipped.
The mascot is an existing NanoLang user-guide asset, copied into this package
as `assets/nanolang-mascot.png`; external artwork is not authority for my claims.
