# My representation constraint graph

I implement the logical APIs of the pinned 599-line `src/nanoisa/nvm2c_shape.c` and its header in pure NanoLang under #1142. This graph derives representation constraints; it grants no decoded-module, type, ownership, capability, profile or executable authority. My replacement order requires the whole graph/solver before complete module/callable/body composition.

## My exact graph

I preserve all 15 private shape kinds: unknown=0, int=1, string=2, array=3, record=4, map=5, optional=6, bool=7, float=8, numeric=9, variant-scalar=10, variant-int-array=11, function=12, u8=13, bstring=14. These are shape kinds, not canonical wire tags (whose bool is 4 and u8 is 2). Exact joins never infer numeric or variant sets from unrelated scalar kinds.

IDs are stable, one-based uint32 values. Zero is invalid. I retain union rank and path compression, projection insertion order, unknown/record permissiveness, array/optional edge 0, map edges 0/1 and exact projection refusals. Lookup does not create a missing node. Producer masks are independent 16-bit annotations. Callable targets are sorted, unique uint32 table indices; an empty set is unresolved, not any target. The caller still validates target indices and signatures against its module.

My NanoLang `Graph` uses a private flat integer arena and a node-address vector. Growth and unification return a `Step` containing the current graph and result number; callers replace their current graph state after each step. Fixed-cell queries may compress paths or poison that state. This mirrors one owned mutable C graph; copying a graph value does not create an independent analysis. A reset returns a fresh empty graph, and old state is discarded. Physical C pointers, allocation capacities and freed backing arrays are not public graph identities. I preserve first logical failure and refuse subsequent operations until reset. Host allocation exhaustion and product publication remain part of the later resource/driver gates; I do not claim allocator-fault equivalence from ordinary semantic tests.

Unification joins roots before queued child pairs so recursive graphs converge without recursive host calls. I preserve kind/projection conflict priority, rank/root choice, producer-mask union and callable/projection order. Failure poisons the graph; partial facts grant no usable result. No C translator, C graph FFI or VM wrapper implements these operations.

## My qualification

A read-only test observer calls the actual unchanged C implementation and reports complete operation outcomes plus abstract graph facts. The production NanoLang graph has no foreign calls. Named expectations supplement differential reports. I preserve the original core recursive/shared tests, 10000-deep success/conflict, 300-wide worklist and all projection/map/poison/reset bounds; I do not replace those with smaller smoke tests. I add all 15x15 exact-kind pairs, rank orders, missing projections, full producer masks and callable sorting/uint32/query refusal cases. Ordinary/sanitized actual C plus C-seed native and both producers’ VM/sanitized-native products retain original 600/60/default10, complete stream/journal/guard evidence and no owned children. Existing C tests run unchanged as a C baseline; their directed-conversion cases do not qualify an unimplemented NanoLang solver.

## My directed solver

I implement ordinary conversion, checked payload, alias view, shared-array alias/write propagation, deferred reads and all fixed-point/final refusal rules in the same NanoLang graph. My [Linux solver qualification](evidence/darwin-current-source-20261124/diagnostics/shape-solver-five-linux-20261126/README.md) passes all 22 methods and 384 commands in 436.235864 seconds. The unchanged C suite passes 3890 assertions and records 232 programs with 132840 expected API operations for replay. All seven products agree, including 791 snapshots per recorded packet. I preserve exact versus inferred storage, canonical optional wrappers, explicit numeric/variant payload sets, cycle/sharing identity, independent producers and one-way callable propagation. All original C conversion/read/write/copy/deep countercases remain required through full NanoLang/native/paired gates before integration or cutover.

My exact core passes its [Linux component qualification](evidence/darwin-current-source-20261124/diagnostics/shape-core-five-linux-20261126/README.md): 18 methods, 279 commands, 287.238492 seconds and 21966 independently verified records. The reader replays every original method with product execution forbidden. My complete directed solver also passes its private Linux component gate. Module representation integration and later platform/release gates remain pending. That earlier graph component changes no compiler/bootstrap/CI selection. The subsequent explicit C handoff and [guarded repaired-candidate Linux receipt](evidence/c-repairs-linux-20261126/README.md) supersede my earlier unanswered-owner barrier: all original private/public/cache, solver and callable core methods pass. I retain the enclosing text-stream observer failure and separately passing companions. My [current-qualified-Stage2 Linux component receipt](evidence/c-repairs-linux-20261126/current-components-20261127/README.md) now verifies the original five solver methods and private/public/cache/callable consumers, retaining the separate whole-observer setup failure; all 69 release requirements remain open where their evidence is incomplete.

My lazy solver arena retains four ordered fact lists and a fixed-point change flag. Alias/write duplicate checks use immutable insertion IDs; later traversal resolves current roots and rebuilds reverse alias adjacency. Writes discovered during traversal enter the same worklist. I retain source/destination separation, inferred-kind markers and the reference pass order: resolve deferred reads, flow every conversion, propagate writes, repeat to stability, then run final producer refusals. I do not claim C hash-table layout or allocator-fault equivalence.

`tests/nvm2c_shape_solver_original_trace.c` includes the unchanged original C tests after wrapping their public API calls. It preserves every original CHECK and records actual evaluated argument/call order. Its binary packet retains each return expectation and callable result. Snapshots after solves and before/after reset compare complete graph facts, inferred-kind flags, ordered conversions/aliases/writes/reads, counts and diagnostics. My NanoLang driver executes these operations itself; it does not invoke the C graph. Original physical-pointer checks remain C baseline checks rather than a NanoLang pointer-identity claim.

`make test-nanolang-translator-shape-solver` adds the recorded original suite, independent directional/final-pass countercases, malformed packets and imported shadows. The existing exact-core and scalar suites remain required. Full module/callable/body/profile/ownership/platform and installed-cutover acceptance remains separate.

## My private storage and carrier facts

My [storage-kind helper](NANOLANG_STORAGE_KINDS.md) passes its exhaustive
131584-input actual-C/five-route Linux component gate. My adjacent
[scalar-carrier helper](NANOLANG_CARRIER_MASKS.md) now passes its65792-input
gate, including exact unknown/mask/optional boundaries, malformed inputs and
imported shadows. Their independent readers retain exact bytes and all
selected guards, with product execution forbidden during replay. Neither
helper grants payload or executable admission. I still compose complete
module representation and body/runtime/profile/ownership/host emission,
qualify paired platforms and installed cutover, reconcile frozen assertions
and pass exact-candidate CI and all69 release requirements.

My [C ABI type selectors](NANOLANG_C_ABI_TYPES.md) now pass their131328-input
branch-domain matrix plus6144 context controls through actual C and five
NanoLang routes. The four-method receipt retains exact spellings/fallbacks,
imported shadows, the missing-header setup failure, unchanged product bytes
and all selected guards. Its reader replays every original method without
products. These facts grant no module/payload/admission authority; checked
table indexing and full module/declaration/body/runtime/profile/ownership/
host composition and all later qualification/release gates remain.

My [complete classifier fact joins](NANOLANG_REPRESENTATION_FACTS.md) now pass
a six-method, seven-route component gate with actual unchanged C. This value
and owned-vector API grants no module/table/payload admission; full composition
and release qualification remain required.
