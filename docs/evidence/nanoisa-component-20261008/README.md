# My NanoISA build component

I replace `transpiler_driver.nano` in my default component build and Stage 3 entry
validation with `nanoisa_driver.nano`. The driver checks ordinary program and
selected-shadow NanoISA emission, including invalid shadow selection. My default
components are now parser, typechecker and NanoISA emitter.

I profile these same components with the C seed's diagnostic `-pg` support. I no
longer hide failed compiler commands behind a `tail` pipeline or ignore failed
entry execution. This diagnostic C-seed route does not add a second self-hosted
product backend.

`qualified.log` passes 19 component/bootstrap controls. The actual
`make test-bootstrap-dependencies` gate passes four component tests plus 16
bootstrap/tool controls. Its real component test compiles the driver with
selected dependency shadows, executes it in NanoVM, translates it to strict C11,
and executes it under ASan/UBSan/LSan. Synthetic Make fixtures check the exact
component drivers, successful stamps and compile/entry failure propagation.

Separate ordinary VM/native runs pass. The real `-pg` build and xctrace run also
return 0 and produce the retained hotspot report. The profiler child completed
before a proposed diagnostic time bound was applied; no process was killed.
`profile-entry.log` separately checks the instrumented binary's program entry
using the profiler's re-entry environment, not a second profiling claim.

The legacy emitter remains available only through explicitly selected historical
drivers/tests. Removing all remaining legacy source/dependency references and
renaming the old compiler phase remain parent retirement work; this checkpoint
closes the default component dependency only.
