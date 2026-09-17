# My declaration ownership scan

I classify global declarations once, before lowering their initializers.
A bool array starts with every declaration marked global. Ordinary and
unsafe block statements clear their let IDs, and function parameter ranges
clear their IDs. I then visit declarations in the same order as before.
Names, module owners, IDs, duplicate checks and initializer lowering are
unchanged. The prior query remains as an independent shadow-test oracle.

My shadow compares every flag for globals, parameters, local declarations,
nested blocks, unsafe blocks and shadow-local declarations. My permanent
Python regression compiles the emitter itself to bytecode, compares its
assembly bytes with the native-built emitter, and executes a program that
checks initialization order and local shadowing.

Validation:

- The existing `make test-nanoisa-src-nano` gate passes 86 comparison checks
  and 63 Python methods. The additional VM-emitter ownership method passes
  separately (3.143 seconds); all 64 methods were exercised.
- I compile both old and new emitter sources with the same C-seed NanoVirt
  executable, then execute both with the same `nano_vm` on a generated
  400-worker fixture. It contains 4,003 explicit let declarations plus
  parameters, nested/unsafe scopes and shadow declarations.
- One retained measurement is 14.18 seconds before and 0.86 seconds after.
  Both runs exit zero and produce the same 205,847 assembly bytes, SHA-256
  `52def4613047b955884967eb41ddc4fbd33e811261075362a473e10831ee430d`.
  This is a measured fixture result, not a general runtime speed guarantee.
- A freshly built canonical compiler publishes the full compiler as
  `/tmp/nanolang-ownership-full.nvm` (350,864 bytes), which verifies in my VM.

Logs: `/tmp/nanolang-declaration-ownership-gate.log`,
`/tmp/nanolang-declaration-ownership-vm-gate.log`,
`/tmp/nanolang-ownership-large-{before,after}-time.log`, and
`/tmp/nanolang-ownership-full-build.log`.

This closes the bounded emitter optimization within
`task_36ceaa830d7d46ba8a5471326f525aac`. Full VM compiler-source execution and
fixed-point comparison remain open. At this checkpoint native-shadow C generation still had its own
declaration-ownership query. My subsequent shared classification is recorded
in `shared-declaration-ownership.md`; native root-tracing work belongs to a
separate runtime follow-up.
