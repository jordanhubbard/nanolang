# My exact depth-two integer-array authority receipt

I qualify the bounded contract in `docs/NANOISA_NESTED_INT_ARRAY_AUTHORITY.md` on
Darwin, from isolated base `0f9ca2300`. I do not claim fresh raw selfhost generations,
Linux qualification, arbitrary nested arrays or completion of #1001.

## Implementation and boundaries

I preserve the original v3 ARRAY_FIELDS descriptor identity and independently
normalize only ARRAY -> INT into a nested obligation. Whole-flow closure checks
both surviving children and all writes, including removed/replaced aliases.
Flat-only work/storage budgets retain their old boundaries. Copied plans and
public transport consumers independently validate the descriptor and flow.

I qualify actual public VM and the separate private VM, including duplicate
child retain headroom, zero-owner teardown, bounds errors, failure recovery and
acyclic CALL/TAIL forwarding. Canonical native records retain shared inner-array
identity under forced collection. I corrected a native forward-result ordering
bug by deferring unknown canonical array carriers until callee facts converge.

Both source publishers retain exact checked declaration identity and whole-module
staging. Only exact array<array<int>> fields gain this authority. Nested strings,
depth three and foreign bodies still omit optional metadata. ARR_REMOVE remains
outside the existing selected profile; its initial rejected probe is retained as
an explicit control, not reported as successful nested support.

## Focused evidence

- `source-expanded.log`: 16 source methods pass for the C producer and a freshly
  C-hosted Nano emitter, including metadata, actual VM/native execution, aliases,
  generic function instances, empty fields, record DAGs and unsupported omission.
  Imported cases use the canonical bound producer selection documented separately.
- The same log retains six process/allocator methods: 264 C publication and 259
  host-query flat allocation positions; 268 positions each for nested publication
  and nested host query. Each position runs transient and persistent failure with
  reached-injection checks, cleanup, sentinel preservation and valid recovery.
- The origins gate passes 1,400 checks, including exact flat and nested work-budget
  limits, incorrect child descriptors and writes through no-longer-surviving aliases.
- `final-regressions.log`: tail gate passes (25.617 s); private/public nested gate
  passes (31.330 s), including six positive fixture families and 24 forced-GC
  native executions across O0/O2 and the two runtime builds. Proof/public VM/private
  preparation/private execution prefix counts are 54/279/85/8 respectively,
  each tested transiently and persistently with recovery. Native identity (2),
  native nested arrays (3), and declaration provenance (9) methods also pass.
- `regressions.log`: existing descriptor (1), declaration projection (1), copied
  execution (1), private VM (2) and public lifecycle/allocation (1; 222.142 s) gates
  pass. Its later tail failure is retained and superseded by the final log.

The tail regression exposed my new retain preflight using initialized
`global_count` instead of reserved `global_capacity`. Existing LOAD_GLOBAL permits
a reserved void slot before initialization; the corrected check preserves that
behavior and the shared handler's out-of-bounds result. `tail-debug.log` retains
the opcode/status evidence. I did not weaken the tail fixture.

The source query harness now explicitly builds its real shared query provider;
a native probe can link statically and does not guarantee an incidental module
cache library. This is a test prerequisite correction, not mocked host evidence.

Log copies omit compiler/link command lines longer than 1,800 characters.
`original-log-sha256.txt` records hashes and retained paths of complete originals.

## Reproduction

I run the bounded consumer suite with `make -j1 test-record-array-nested` and
origin/descriptor gates with `make -j1 test-record-array-origins
test-ordinary-array-authority`. I run paired source and allocation coverage with
`make -j1 test-flat-array-source-publication test-publication-allocation`.
For imported canonical-source coverage I build `src_nano/nanoc_v06.nano` using
`bin/nanoc_c`, then provide that binary and `bin/nano_virt` through
`NANO_FLAT_ARRAY_BOUND_COMPILERS`. The ordinary emitter selection remains
`NANO_FLAT_ARRAY_COMPILERS`. Both variables are comma-separated executable paths.
The canonical product is C-hosted; I do not relabel it as a fresh raw Stage1/2.

My final `final-publication.log` rerun passes all 16 producer methods (16.079 s),
now including imported flat and nested records through both the C producer and
fresh C-hosted canonical driver. The final consumer rerun passes both builds
(30.660 s) with an added decoded-transport forgery/recovery check: changing the
retained leaf from INT to STRING refuses conversion, restoring INT independently
rederives a valid executable module. This does not bypass container checksums;
it directly tests the decoded module conversion boundary.
