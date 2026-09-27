# Constructor-indexed native shape foundation

I add `NVM_SHAPE_VARIANT` to the private native constraint graph. Its edges are uint16 constructor tags, each retaining a separate payload shape. Native integration will place each constructor's complete record of fields under that tag. Directed copies accumulate distinct tags without equating their payload layouts. Conflicting types under the same tag still fail. A consumer does not invent producer tags, and this shape cannot unify or convert directly to an ordinary record.

My additive tests exercise both producer orders, deferred local/return copies, distinct record fields and nested string arrays, the maximum tag and an out-of-range tag, exact same-tag conflicts, unknown producer evidence, cyclic payloads, shared children, and separately constrained destination views. Existing scalar/integer-array variant and negative shape tests remain intact.

The first run found a copy defect: two references to a shared array became different nodes when their conversion was still pending. I now reuse a freshly created pending target before its kind is populated. Existing explicit destination views remain distinct. The original failing run is retained.

This is a graph foundation, not a native union implementation. `nvm2c.c` still classifies variant field storage through its flat field vectors and cannot translate the six retained source subtests. I leave `task_065a1a2c9858b8968fd3851135b48c47`, aggregate formatting and PR522 open.

## Integration still required

I must carry constructor-indexed facts through `AGG_PACK`, local/global stores, stack joins, call parameters and returned aggregates. `shape_kind`, `shape_record_return` and projection classification currently impose `RECORD`; they must distinguish a union's constructor map from an ordinary record's field map. Flat `rec_k` vectors cannot remain the authority for heterogeneous variant payloads.

I must retain the relation between `AGG_TAG`, its source value, a comparison and each conditional edge. A checked constructor branch selects that constructor's payload shape. I must invalidate stale relations after assignments and preserve only agreeing facts at joins and loop edges. Unknown tags and missing producer evidence do not authorize arbitrary record projection. A shape ID identifies storage constraints, not a particular value: a tag test before reassignment cannot narrow a later load merely because both loads share that shape. I need value provenance on each edge, including stack aliases and local invalidation; the current single-tag summaries lose this distinction after mixed assignments.

I must make emitted storage/projection checks agree with those facts. The existing `nrec_t` runtime already stores record snapshots and per-field runtime tags, but that alone does not establish safe heterogeneous classification. I must retain managed roots, copy and lifetime behavior and the existing wrong-tag/layout negative controls. Resource-bearing payloads require their separate ownership contract.

I will use the unchanged five-method aggregate-formatting matrix, added raw native controls, and complete ordinary/instrumented translator gates to qualify integration. Shape tests alone cannot close the native storage task.

## Qualification

`make test-nvm2c` passes all 2,657 translator assertions and 1,715 shape checks. A fresh private translator build passes the same counts under ASan/UBSan, leak detection and stack-use-after-return detection; the runner verifies instrumentation in the actual translator and shape objects. The standalone strict-C11 instrumented shape binary also passes all 1,715 checks.

The original graph run retained in `first.log` has 1,706 passing checks and one failing shared-child assertion. `corrected.log` passes 1,707 checks before the final eight explicit-view checks were added. The final counts above include those eight checks.

The five-method source matrix takes 8.512 seconds and retains exactly six native failures. C-seed and both VM routes pass all methods; native array and retained-record string methods pass. I do not claim native heterogeneous union support from these graph results. No assertion, deadline or instrumentation setting was weakened.
