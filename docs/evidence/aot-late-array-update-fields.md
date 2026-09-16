# My late record-array update facts

My compiler reached `ARR_SET` in function 191 at offset 83 with field 4
unknown in the destination's flat facts and string in the replacement's
facts. My final traversal treated that as a known incompatibility even though
recursive constraints were still being collected.

I reject differing flat field representations only when both are known.
I still unify the destination element shape with the replacement shape, so
later contradictory facts fail. My generated code still checks index bounds,
record width, aggregate kind and each stored field kind before replacement.

My new tests project a record array from an enclosing record, update it in a
callee, then observe the change through the caller's original array handle.
Both function orders run. Replacing the nested string field with an integer
is rejected in both orders. Existing bounds and layout rejection tests remain.

I run `make -j1 test-nvm2c`, `make test-nvm2c-sanitizers` and
`make -j1 test-one-ir-compiler`. Full compiler acceptance is incomplete: it
passes the former update rejection and reaches boolean-array append in
`typecheck_local_lets` (305), offset 41. I track that under
`task_4fb63e3486c4404fa2f8b9e5ee8124a9`. The focused empty-array source fixture
passes. Shape errors now report their function and bytecode offset.

Normal and fresh ASan/UBSan suites pass 1,534 AOT and 994 shape checks each.
Leak detection is disabled; I do not infer leak freedom from these runs.

MAC rejected my task claim with `agent_status_unavailable`. A separate
`mac task why-unclaimed` check reports `task_project_inactive` and no eligible
candidates among 27 listed workers. I leave that project-wide setting unchanged.
