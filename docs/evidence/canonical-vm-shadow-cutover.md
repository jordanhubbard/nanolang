# My canonical bytecode shadow cutover

My proposed `--emit-nvm` route lowers the checked bound parser into production
and selected-shadow modules. It verifies the staged shadow module, invokes
`nano_vm --check-shadows`, cleans temporary storage, and publishes verified
production bytes only after successful completion. This route returns before
`transpile_parser` or native shadow compilation. My default native product
path remains separate architecture work.

`NANO_VM` selects an explicit runner; otherwise I use `bin/nano_vm` under my
discovered repository root. I quote both the runner and staged module paths.
My existing parent deadline configuration applies inside that runner.

A fresh native bootstrap passes. Five focused methods pass through both
Stage 1 and Stage 2 with a rejecting native-compiler stub: shared global
state and ordered shadows, ordinary user main, dependency/root-only selection,
assertion and deadline failures, previous-output preservation, missing runner,
no-shadow compilation, and runner paths containing spaces.

After the filesystem artifact repair in PR #499, my unchanged canonical
publication gate still has two failing methods out of eight. File assembly
shadows now reach the undeclared lowering contract `nl_nanoisa_assemble_save`;
an unused helper's floating-point shadow also remains unsupported. I retain
those tests and tasks `task_f5f873fccfff4b5b88f14f4d825ba3b4` and
`task_8bc58d33e73f4e24a474d3724d862c94`. The full compiler shadow closure
also retains its recorded range-loop prerequisite. This cutover remains a
draft until its prerequisites and unchanged acceptance checks pass.

Logs are `/tmp/nanolang-canonical-vm-shadow-fs-{bootstrap,integrated}.log`.
That integrated run contains 13 methods and records two failures in 8.684s.
This is `task_c5a7a4835d364b50b747018c794a07d0`; I do not claim the full
NanoISA-only bootstrap or release acceptance yet.
