# Nested record values

I snapshot a nested record into immutable heap storage when packing it into
another record. This storage outlasts the creating function. Extracting the
nested record copies its value into a new temporary; later constructions do not
overwrite an earlier snapshot. Arrays inside records retain their existing
reference semantics.

Generated programs own snapshots in a linked list and release them iteratively
after the entry function returns. This avoids recursive destruction and leaves
no nested-record pointer referring into a returned function's stack. I do not
claim bounded retention during long-running execution: unreachable snapshots
remain until entry returns. I filed reclamation as
`task_d152cc3913f248fb8d1483210e60f00b` and added it to the roadmap.

Nested projections retain unknown field facts rather than guessing integer
fields. After the final graph constraints are collected, resolved local kinds
select storage for values such as strings extracted into locals. The graph
also supplies field representations when extracting nested records.

## Verification

`make -j1 test-nvm2c-sanitizers` passes 1,057 AOT checks and 965 graph checks,
with fresh ASan/UBSan object instrumentation verified. Tests extract nested
fields directly and keep a three-level returned record alive across a second
constructor call, checking both integer and string fields. The string passes
through a local before use. Existing strict generated-C compilation remains on.
A subsequent ordinary `make -j1 test-nvm2c` passes the same check counts.

The first run found an unused snapshot-helper warning in flat-only programs;
I fixed its generated reference without suppressing warnings. The deeper
regression exposed integer assumptions for unknown nested fields; those fields
now remain unknown until resolved.

`make -j1 test-one-ir-compiler` no longer stops at function 100's unsupported
nested packing. It now fails with conflicting parameter or aggregate-field
inference. That gate remains open, as does general recursive inference.
`git diff --check` passes.

Parent MAC task: `task_9c850e94e5a74b6f8941622e2872af23`.
