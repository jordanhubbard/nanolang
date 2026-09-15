# My self-hosted scalar map lowering

I replace the int-only fallback for known scalar transforms with typed
call-site lowering. `generate_scalar_map` derives input and output types from
the transform's signature, including named functions, function variables and
returned function expressions. Source evaluation precedes transform
evaluation, and each happens once. I allocate result storage from the result
type even when the source is empty. Input reads and result pushes use their
respective typed array helpers; I do not cast callback signatures.

I leave unsupported signatures on the existing path rather than claim
aggregate support. This patch does not complete checker validation, nominal
or nested metadata, aggregate layout, closure capture semantics, or resource
ownership. MAC `task_75b340982b6cf797f29b38c1a188aab3` remains open.

## Verification on Darwin, 2026-09-15

`make test-selfhost-map-results test-selfhost-returned-calls` rebuilds both
compiler stages and passes their smoke/no-C-seed checks. It then passes all
16 scalar map pairings in 44.385 seconds and all three returned-call methods
in 8.321 seconds. Log: `/tmp/nanolang-selfhost-scalar-map.log`.

The existing map matrix retains named, variable and returned transforms,
two-element and empty sources, result indexing, input preservation and
append-after-empty. Each fixture compiles with default shadows and executes
the published artifact. Before this change, 15 pairings failed native
compilation against the int-only callback signature.

I add a Stage2-specific trace with a computed source and returned transform.
Its exact stdout is `source\nchoose\ntransform\ntransform\n`, and assertions
check both float results. The final command
`NANOLANG_MAP_SELFHOST=1 python3 tests/test_map_results.py` passes both
methods (16 pairings plus the trace) in 52.268 seconds.
Log: `/tmp/nanolang-selfhost-map-final.log`.

The trace is selected by the self-hosted gate; the C-seed/VM matrix target
does not acquire an undeclared Stage2 dependency. I did not rerun the full
release suite. MAC rejected my claim with `agent_status_unavailable`; passing
this scalar checkpoint does not close the broader ledger task.
