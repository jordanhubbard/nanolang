# My PR #294 reconciliation

Main merged PR #294 as `5df69d985de9601c5d15001e092bc116197aef94`.
The worker head is `6590e747a04bb641e15edf1956b2489c065ed162`. Their complete
trees match: `ba0deba677da42ab719e1e01deb47e3d354f74da`.

I combine that work with integration's `2c771d3a` nested-array repair. I retain
the newer parser element spans, recursive checker metadata, literal child
types, empty-storage fix and declared/computed function-call inference.

From PR #294 I retain runtime helper selection for full nested-array types,
its incremental cube fixture and its self-hosted runner entry. I add a useful
shadow to the fixture and exercise it through the focused Python gate as well
as retaining the shell-runner entry. The shell runner alone is not my evidence
for negative diagnostic boundaries.

I use the existing exact outer-wrapper extractor for
`array_elem_type_from_array_type`; I do not retain the old substring fallback
that could select a scalar from inside a nominal or function type. Helpers
for storage tags, reads, pushes, sets and C element types now accept complete
nested-array descriptions. Their shadows check those mappings.

My acceptance command is the serial bootstrap-backed array, map-type,
scalar-map and returned-call gates:

```sh
make test-selfhost-array-compatibility test-selfhost-map-types \
  test-selfhost-map-results test-selfhost-returned-calls
```

The command log is `/tmp/nanolang-pr294-gates.log` on this macOS host.
Both stages rebuilt, their smoke tests and the installed no-C-seed check
passed. The array gate passes six methods in 21.496 seconds; map typing passes
two in 2.929 seconds; all 16 scalar map pairings and their order trace pass
two methods in 48.372 seconds; returned calls pass three in 8.422 seconds.
Native bootstrap binary equality remains a separate unfinished gate.

I retain both the main squash and the identical original worker head as
merge ancestors. The second ancestry-only merge changes no source.

This does not establish full nominal/generic array execution, aggregate maps,
unknown-type soundness or backend equivalence; those remain on my roadmap.
MAC task: `task_f32d71bbad07448c8656843015e72ef3`.
