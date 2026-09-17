# Recursive shape constraint foundation

I need nested element shapes before enabling record-array fields. The current
flat field-kind vectors cannot represent that information. I add a private
constraint graph in `src/nanoisa/nvm2c_shape.c`, with stable IDs, dynamically
allocated sparse field edges, and iterative unification. Record edges denote
field indices; an array's edge zero denotes its element shape.

I unite roots before processing child constraints, so recursive graphs converge
without recursive C calls. Union by rank and path compression preserve shared
identities. Unspecified children are unconstrained, not evidence of absent
fields or a valid runtime field index. Conflicts and allocation failures poison
the graph; callers must discard it. Failed unification is not transactional.

## Verification

`make -j1 test-nvm2c-shapes` passes 952 checks. They cover recursive
record/array cycles, shared children, repeated unification, independent sibling
fields, 10,000-level structures with matching and conflicting leaves, invalid
IDs/projections, and a 300-field merge that grows the worklist. The same test
and graph sources compiled with `-fsanitize=address,undefined`, `-O1 -g` and
`-fno-omit-frame-pointer` pass 952 checks without a sanitizer report.

The new target is a prerequisite of `test-nvm2c`; the existing AOT executable
suite still passes 1,043 checks. `git diff --check` passes.

## Remaining integration

This graph is not yet used by the production classifier or emitter. It does
not establish runtime layout, field bounds, ownership, representation of
recursive values, or compiler acceptance. I must connect persistent shape
variables to instructions, locals, arguments, results and joins, then use the
resolved facts during C emission. Only then can record-array fields be enabled
and the compiler gate rerun meaningfully. I leave that roadmap item and MAC
`task_9c850e94e5a74b6f8941622e2872af23` open. The hub refuses the claim with
`agent_status_unavailable`.
