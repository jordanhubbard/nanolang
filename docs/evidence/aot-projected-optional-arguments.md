# My projected optional arguments

I let a projected scalar argument acquire its shape before solving conversion
into optional parameter storage. An unresolved flat field kind does not imply
that the source shape equals the parameter's optional wrapper. A known string
and an unresolved projection now use the same directed conversion boundary.
Exact incompatible payloads still fail the shape solver.

I traced the `check_expr_node` tail call at offset 397 to argument zero of
`type_unknown_named` (259). The argument had unknown flat kind but a string
shape; the parameter had optional storage. Equating them caused the failure.

My eight regression combinations cover ordinary and tail calls, both function
orders, present and missing projected values, and incompatible integer payloads.
Compatible cases execute in NanoVM and native C compiled with warnings as
errors. Incompatible cases must fail translation with a shape diagnostic.

My normal and fresh ASan/UBSan suites each pass 1,572 AOT and 1,073 shape
checks. Leak detection is disabled; I do not infer leak freedom. Compiler
acceptance runs five tests: four focused tests pass and the full compiler fails
at the later boundary below.

Full compiler acceptance remains open. It passes the former tail-call conflict
and now stops at `ARR_PUSH` in `cg_append` (359), offset 6, with string/optional
shapes. Its disassembly declares array and string parameters: this is a scalar
string-array write, not a record-array join. I track it in MAC
`task_ed5f20e9b78d4759b44e7b496b92a2ea`.

MAC refuses my claim for `task_5c9df665e6024af3a0cd243dfa6fe8ca` with
`agent_status_unavailable`. I retain evidence without forcing a closure.
