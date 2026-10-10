# My shared File reference batch

I continue #989 from `d11b7c0c6`. Shared helper parameters retain mode 1 through
both source lowerers, nominal declarations, logical facts and runtime formals.
Aliased shared roots use one frozen host lease. I promote another root when
its origin ends first, retain the epoch, and release the lease after the final
root and formal aliases end. My File methods still require exclusive access.

## Qualification

- `foundations.log`: linked and instrumented flow checks pass; the two private
  carrier lifetime methods pass, including shared origin promotion and refusals.
- `runtime-sanitize.log`: the two private carrier methods pass with ASan/UBSan
  in 75.100 seconds. Their selected providers are instrumented; unrelated linked
  compiler/common objects are not an instrumented whole compiler.
- `source.log`: my first source checks fail at nominal analysis, revealing a
  remaining exclusive-only parameter descriptor restriction.
- `source-corrected.log`: three focused C lowering methods pass in 73.234 seconds
  after admitting shared descriptors and checking legal shared map aliases.
  They compare granted VM/native results, shared assertion cleanup and limits.
- `cli.log`: my shared fixture passes both C drivers, including all ten shadows,
  VM/native execution and mixed-mode calls on distinct owners. The refusal
  test rejects the intended source, but initially expects the wrong diagnostic.
- `refusals-corrected.log`: the corrected refusal expectation passes both drivers.
- `nano-source.log`: the Nano probe fails during dependency-shadow setup at its
  unchanged ten-second deadline; no Nano source methods execute in that run.
- `nano-traced.log`: unchanged dependency-shadow selection completes on the next
  traced run, but the positive case reveals my probe selected `shadow main`.
  The shared assertion-cleanup method passes. This does not establish the cause
  of the earlier deadline.
- `nano-corrected.log`: both independent Nano source methods pass in 98.977
  seconds after reserving `main` for the program. Both probe forms emit identical
  bytes; granted VM/native consumers agree on the result and assertion cleanup.
- `shadow-failure.log`: both C drivers preserve prior output when a selected
  shadow fails with multiple shared aliases live.
- `neighbors.log`: linked/instrumented nominal, flow, cyclic, cyclic-hosted and
  indirect-hosted tests pass, including copied facts and allocation-failure
  sweeps. These query tests do not grant runtime or publication authority.
- `make-diagnosis.log`: a dry-run trace locates an observed startup delay at
  recursive dependency-file discovery under `obj`; it does not establish why
  that filesystem traversal was slow.

I also retain `nano-cli-initial.log`: the unoptimized publication artifact hits
my unchanged ten-second File shadow deadline in both the new shared suite and
the existing exclusive suite. Its module metadata previously had no optimization
flag; I now build it with `-O2`. I keep all selected shadows and the same deadline.
`optimized-artifact-hashes.log` confirms a new build context and 254 project
source/dependency hashes matching the worktree. `log-flush-timing.log` measures
24 flushed-and-synced shadow-sized records at about 11 ms on this machine.

The optimized compiler rebuild twice hit its separate dependency-shadow deadline
(`optimized-driver-first.log`, `optimized-driver-retry.log`). I add opt-in C-seed
VM markers to the existing `NANO_SHADOW_TRACE` path. The subsequent traced build
completes all selected shadows (`optimized-driver-traced.log`); I do not claim
that tracing fixes or explains the earlier deadlines. The two trace tests check
opt-in output and timeout attribution with prior-output preservation. Two
neighbor tests retain import order, product separation and failed-shadow refusal.

`optimized-nano-cli.log` passes all four selected driver methods in 239.156
seconds, using both VM and native compiler forms. I execute the complete shared
and existing exclusive shadow suites, compare emitted bytes and granted
VM/native results, and retain refusal and failed-shadow output preservation.
The complete shared suite selects ten shadows, including all five unchanged
generated binding shadows. This is a focused four-method driver matrix, not a
claim that every inherited driver test ran again.

I do not claim installed-platform, release fixed-point or 5.1 publication gates from this development
batch. Indirect File source calls, multiple nominal catalogs, the full mixed
profile, compiler/backend, Socket/network and exact release gates remain open.
