# My verifier profiles

I select a profile through a consuming API, not through advisory module metadata.
This bounded prerequisite does not complete my compute-profile or target-coverage
roadmap. MAC `task_037b12aecc894b86ba335828fa1eb1a2`.

## My admission contract

- `NVM_PROFILE_GENERAL` returns exactly my ordinary `nvm_verify` decision.
- `NVM_PROFILE_CLOSED_SCALAR` first performs ordinary verification, then applies
  the LLVM translator's shared module, signature and opcode checks. My initial
  extraction preserved eligibility unchanged; later matched lowering extends
  the explicit opcode list. Generic numeric arithmetic is specified in
  `NANOISA_LLVM_GENERIC_NUMERIC.md`.
  I require an explicit zero-argument integer/bool entry. I refuse imports,
  module references, nominal declarations, retained layouts, ownership/passive
  contracts and captures. My scalar global/initializer extension follows
  `NANOISA_LLVM_SCALAR_GLOBALS.md`. I retain existing numeric,
  bool and void signatures and the explicit existing instruction whitelist.
- `NVM_PROFILE_CLOSED_LITERAL_STRINGS` retains the scalar module rules and adds
  literal byte transport and string signatures, STR_LEN and STR_EQ. It refuses
  ADD/CAST_INT/CAST_FLOAT anywhere in a string-bearing module. The existing
  CLOSED_SCALAR selector continues to refuse strings. Details and lifetime are
  in `NANOISA_LLVM_LITERAL_STRINGS.md`.
- I reject unknown profile selectors. Neither a source annotation nor arbitrary
  metadata can select or bypass the consuming tool's profile.
- LLVM selects CLOSED_LITERAL_STRINGS through this shared admission. Wasm uses the same LLVM route. My initial
  extraction changed no target eligibility; later extensions require matched
  lowering and their documented gates. I retain serialization and VM default
  admission. This scalar profile is not a GPU kernel contract.

## My acceptance

I compare general and restricted API decisions with translator decisions on
ordinary admitted scalar modules and normal out-of-profile programs. I preserve
an existing output artifact when LLVM or Wasm refuses a module. I rerun the
existing integer, float, U8, truthiness and implicit-return translator gates.
Unknown profiles refuse; ordinary verification failures remain failures.
My complete applicable-language LLVM/Wasm coverage remains open.
