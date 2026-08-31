# NanoLang 3.5

I am NanoLang 3.5. This release closes my NanoISA measurement and cleanup
foundation. I keep the remaining NanoISA v2 completion work scoped to 4.0.

## What I Shipped

- I record repeatable NanoISA profiles for seven maintained workloads.
- I measure retired instructions, opcode sequences, branches, calls, stack and
  frame depth, traps, heap traffic, and FFI traffic.
- I dispatch predecoded NanoISA instructions in NanoVM.
- I support direct tail calls with verifier and runtime signature checks.
- I generate active ISA metadata from `spec/nanoisa.yaml`.
- I keep generated source locations in side tables rather than executable
  debug instructions.
- I provide optional opcode diagnostics through `NANO_VM_TRACE`.
- I provide runtime-selectable generated-C profiling through `NANO_PROFILE`.

## Evidence

I verified the release with:

- NanoISA: 1,077 tests passed.
- NanoVM: 322 tests passed.
- NanoVirt: 63 tests passed.
- Seven NanoVM benchmark workloads, 20 samples each, all exit codes `0`.
- CI checks across x64, arm64, C, PTX, RISC-V, sanitizers, coverage,
  documentation, benchmarks, and security.

## Boundary

I have not called my NanoISA v2 verifier, module-linking redesign, typed FFI
ABI, runtime representation changes, or superinstruction work complete. Those
items are documented as 4.0 work in [my roadmap](ROADMAP.md).

## Links

- [Current README](https://github.com/jordanhubbard/nanolang/blob/main/README.md)
- [3.5 changelog entry](../CHANGELOG.md#350)
- [3.5 GitHub release](https://github.com/jordanhubbard/nanolang/releases/tag/v3.5.0)
- [NanoISA design and evidence](NANOISA.md)
- [Performance monitoring](PERFORMANCE_MONITORING.md)
- [NanoVM opcode debugging skill](../skills/nanovm-opcode-debugging/SKILL.md)
