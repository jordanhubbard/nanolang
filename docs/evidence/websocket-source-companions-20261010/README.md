# My owned WebSocket source companions

I build on `fa9569a02`. I add an exact WebSocket binding provider and explicit
catalog-3 selection in the shared immutable companion snapshot loader. I include
that provider in compiler input archives and the `file_source_inputs` module.
I do not yet register WebSocket source catalog queries, namespaces or lowering.

My first fixture compile exposes a mechanical spelling error in its opaque type;
I correct the fixture. The next run reaches valid document preparation and fails
its canonical roundtrip. The shared renderer writes only `capabilities[0]`, so
WebSocket loses its separate lookup permission. I record the defect before
repair and render every declared capability. File/TCP exact output stays unchanged.

My final checks:

- LLVM ASan/UBSan/LSan: three WebSocket methods pass in 17.746 seconds, including
  all 798 document cases, allocation-failure prefixes, immutable snapshots and
  the existing NSI/generator/WebSocket catalog neighbors.
- GCC: the same three methods pass in 11.229 seconds.
- LLVM sanitizers: all five File/TCP binding methods pass in 33.255 seconds,
  including exact canonical/source byte comparisons and allocation failures.
- Actual module consumers: C seed native, NanoVirt VM/native, Stage 1 VM/native
  and Stage 2 VM/native pass in 16.104 seconds. They acquire all three catalog
  companions, reject mismatches and retain copied WebSocket strings after free.
  Each compiler runs its selected shadows. This tests the input bridge, not
  WebSocket service source lowering or a new compiler fixed point.
- Actual Make `test-c-service-inputs`: three original loader/origin/input methods
  pass after rebuilding the compiler input archive and C/NanoVirt tools.

My binding receipts retain exact commands, terminals and stdout/stderr. The
snapshot test deletes the opened companion, preserves canonical/declaration
views and verifies failed acquisitions do not publish indices or change retained
storage. Its context stays within the existing 64 MiB bound.

Public WebSocket real-peer qualification is still blocked by this sandbox's
localhost-bind restriction. Paired source catalog/namespace/lowering work,
public DNS service integration and final release/platform gates remain open.
