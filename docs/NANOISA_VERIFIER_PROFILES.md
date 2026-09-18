# My verifier profiles

I select a profile through a consuming API, not through advisory module metadata.
This bounded prerequisite does not complete my compute-profile or target-coverage
roadmap. MAC `task_037b12aecc894b86ba335828fa1eb1a2`.

## My admission contract

- `NVM_PROFILE_GENERAL` returns exactly my ordinary `nvm_verify` decision.
- `NVM_PROFILE_CLOSED_SCALAR` first performs ordinary verification, then applies
  the existing LLVM translator's module, signature and opcode checks unchanged.
  I require an explicit zero-argument integer/bool entry. I refuse imports,
  module references, nominal declarations, retained layouts, ownership/passive
  contracts, captures and initializer functions. I retain existing numeric,
  bool and void signatures and the explicit existing instruction whitelist.
- I reject unknown profile selectors. Neither a source annotation nor arbitrary
  metadata can select or bypass the consuming tool's profile.
- LLVM uses this shared admission. Wasm uses the same LLVM route. I change no
  serialization, VM default admission, runtime semantics or target eligibility.
  This scalar profile is not a GPU kernel contract.

## My acceptance

I compare general and restricted API decisions with translator decisions on
ordinary admitted scalar modules and normal out-of-profile programs. I preserve
an existing output artifact when LLVM or Wasm refuses a module. I rerun the
existing integer, float, U8, truthiness and implicit-return translator gates.
Unknown profiles refuse; ordinary verification failures remain failures.
My complete applicable-language LLVM/Wasm coverage remains open.
