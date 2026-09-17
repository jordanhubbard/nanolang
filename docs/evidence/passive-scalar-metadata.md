# My scalar passive metadata boundary

I add a versioned v2 record for constant/scalar `par` and `flow` graphs, with
canonical source IDs, exact dependencies and verified stable serial ranges.
The record is copied through the runtime bridge, retained byte-for-byte across
v2 round trips, and refused by legacy serialization. The ordinary VM and native
translator execute the unchanged serial code. No frontend emits these records
yet; this is the IR prerequisite, not completion of passive parallelism.

`make test-passive-metadata` passes 204 checks plus execution of one serialized
forward-dependency graph in NanoVM and strict C11 native output, both printing
42. Checks cover independent nodes, a forward dependency, duplicate or cyclic
claims, wrong ranges/results/effects/resources, every truncated prefix,
rejected external inputs and semantic rejection after a corrected container
checksum. Repeated serialization preserves the exact bytes. The record checker
and test driver also pass those 204 checks with ASan and UBSan instrumentation;
the linked existing support objects in that run are not instrumented.

Existing NanoISA tests pass 2,691 checks. Container, whole-module, bridge,
end-to-end and verifier gates pass, as do five wrapper-generation checks and
seven wrapper publication tests. A second agent reviewed the record bounds,
graph/range validation and payload ownership and found no concrete blocker.

External input reads remain refused. Ordinary type verification widens local
and call values to unknown, so a declared parameter tag is not a verified proof
that a caller supplied an immutable scalar. Task
`task_bf571298c10d4cc5a387b9f233ff3c40` owns that continuation. Calls, captured
values, aggregates, resources and host operations remain outside this first
record version. Full frontend/metadata equivalence, richer closed calls and
all passive conformance rows remain open. This bounded IR slice is tracked by
`task_abd0941bc1f34580845eb31ecd68a12c`.
