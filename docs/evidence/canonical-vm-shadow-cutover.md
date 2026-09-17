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

After the filesystem and file-facade artifact repairs in PRs #499 and #503,
my unchanged canonical publication gate has one failing method out of eight:
an unused helper's floating-point shadow remains unsupported. I retain that
test and `task_8bc58d33e73f4e24a474d3724d862c94`. The complete compiler shadow
closure has additional recorded prerequisites; passing these focused tests
will not by itself establish its acceptance.

At main `54287161` plus this cutover, a fresh native bootstrap passes with
the supported explicit 60-second shadow budget. The combined 13-method gate
passes 12 methods and records the floating-point failure in 8.905 seconds,
without a budget override. Logs are
`/tmp/nanolang-canonical-facade-{bootstrap,integrated}.log`.
This is `task_c5a7a4835d364b50b747018c794a07d0`; I do not claim the full
NanoISA-only bootstrap or release acceptance yet.
