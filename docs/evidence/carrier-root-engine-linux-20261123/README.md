# I account for carrier graphs before adapting source ownership

I qualify my private engine at source
`c6eb52f54805d0a35f9adbf23c9f5e764cdf9759` under #1211/#1116/#1140.
I add src/eval/carrier_heap.c/.h and two focused tests to ordinary units.
I do not yet link the engine into the evaluator. Existing evaluator,
environment, GC, source admission, container layouts and compiler/VM/AOT
images remain unchanged. The actual typed-list source UAF and both previous
full source leak reports remain open.

My engine owns only explicitly registered allocations with supplied destruction
policy. Its private pointer index and iterative marking queue preserve shared
and cyclic reachable identities and reclaim unreachable cycles exactly once.
Unknown roots/edges grant no ownership. I reject duplicate registration,
missing policy and mutation/reentrant collection during callbacks. Explicit
forget transfers one allocation without destroying it; the adapter must account
for every owned descendant crossing a host/interpreter boundary and remove
entries before early manual discard. Destructors retire their own storage and
scalars, not registered children. This is a per-interpreter engine; source
producers still have to establish actual ownership and enumerate actual roots.

Fresh strict GCC13 Linux ARM64 ordinary and ASan/UBSan builds both pass the
complete focused target with leak detection enabled. My graph fixture balances
67,664 payload objects and their separately owned text buffers, checking:

- explicit replacement/list roots without copying retained storage;
- shared edges and reachable/unreachable cycles;
- caller-owned unknown storage, explicit transfer, early discard and readoption;
- ownership transfer between separate heaps and reentrant mutation refusals;
- a65,536-object linked graph and repeated allocation/removal cohorts.

My second fixture compiles the actual engine with only its metadata allocator
names redirected. It injects six actual failures: both heap-creation stages,
entry creation, mark-queue creation, table growth and entry creation after
successful growth. Caller payloads remain valid, registry ownership/statistics
stay intact and every metadata allocation is released, including teardown
while further allocation is disabled. Both profiles have empty stderr.

I retain the two preceding successful development cohorts and their products.
The second cohort's output calls six injected failures "five"; I correct only
that message in the final fault fixture, with every assertion unchanged. The
first Make recipe is reconstructed and checked against its contemporaneous
recorded SHA after I append the fault fixture. Earlier graph source is also
verified against its recorded SHA. Final qualification adds the actual cross-
heap transfer control; it removes no earlier assertion or case.

I verify my source/products/archive with:

```sh
python3 docs/evidence/carrier-root-engine-linux-20261123/read_archive.py
```

My reader passes1,437 records/1,398 exact Git inputs and ten actual native
ARM64 products. It verifies compressed/raw hashes, object coverage, commands,
complete streams, earlier cohorts, exact Make-only addition, unchanged existing
source/test/runtime/tool inputs, current GCC13.3/as2.42 identities and the
hash-bound preceding archive. Identical objects remain shared with those
archives. This is engine qualification, not a fresh complete interpreter build,
Darwin/current-source bootstrap, generic toolchain or release acceptance.

My next required step is the evaluator adapter: register actual fresh source
carriers and manual clones, enumerate binding/evaluation/argument/aggregate/
typed-list/returned/async roots, coordinate early discard and GC lifetime, then
collect at safe boundaries and qualify the original complete source/platform
gates. I retain the [actual red source/API witness](../carrier-root-failure-linux-20261123/README.md)
and previous1937/82 and3720/159 full LSan reports. I do not substitute this
unit graph for source-pipeline acceptance.

Puck retains #1204/#1190/nvm2c/merge/regex/capture/checkpoint/CI. All frozen
roots/audits/stashes and NAS/TCC/migration pauses remain intact. My serial
Linux native cohort is terminal; no Darwin reservation or NAS action.
