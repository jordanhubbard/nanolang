# I propose read-only ordinary-heap map import validation

This is the next review checkpoint after boundary-summary component0d8232c1f.
It is not implementation or public execution admission. I preserve the existing
private runtime, public VM/native selection, loaders and publishers. The first
pair remains STRING→INT; DELETE is outside the selected profile.

## Actual source interfaces

- `managed_record_map_boundary.inc` owns one copied execution plan and historical
  global rows from the same successful proof. Its module-taking constructor and
  immutable queries are qualified; it knows no runtime object addresses.
- `heap.h` exposes ordinary managed headers (ref_count, obj_type, colour,
  buffered), intern buckets and the cycle suspect buffer. There is no allocation
  registry or object-to-heap identifier that can establish arbitrary C pointer
  validity or actual allocation extent.
- `heap.c:vm_release` decrements a live owner. A still-live container becomes a
  suspect; a zero-ref buffered object is marked BLACK and deferred to collection.
- `heap_cycles.c:vm_gc_note_suspect` sets PURPLE before trying buffer growth. An
  allocation failure can leave an unbuffered PURPLE object. Successful insertion
  sets buffered; a second notification does not append an already-buffered object.
- `vm_gc_collect_cycles` takes the old buffer, restores reachable trial-deleted
  counts, normalizes newly buffered roots, removes old buffered flags and performs
  ordinary destruction. It never runs during this proposed validator.
- `record_map_heap.inc:vm_rm_map_header` requires !buffered. Its private validator
  cannot be reused unchanged for ordinary maps. The exact counted hash/probe
  algorithm can be shared only through a private read-only ordinary adapter with
  a separately reviewed header predicate, never a caller-supplied trusted flag.
- `record_map_vm_observe.inc` supplies useful selected-child checks but is not an
  imported-graph validator: it trusts a private owner and rejects buffered values.

## Smallest entrypoint and ownership

I propose a private/internal module-taking query, not an execution selector:

```c
typedef enum {
    VM_RM_IMPORT_OK, VM_RM_IMPORT_INVALID, VM_RM_IMPORT_UNRESOLVED,
    VM_RM_IMPORT_MEMORY, VM_RM_IMPORT_LIMIT
} VmRmImportStatus;
typedef struct {
    uint64_t preparation_bytes, preparation_work;
    uint64_t graph_objects, collector_objects, traversed_edges;
} VmRmImportReport;
VmRmImportStatus vm_record_map_import_check(
    const NvmModule *, const VmState *, VmRmImportReport *out);
```

This function constructs and owns a fresh boundary plan, then calls a static
borrow-only validator in the same implementation TU. No exported helper accepts
an allegedly trusted plan or skips the module proof. The static helper borrows
that plan, caller state and a checked local ledger; it cannot return a plan,
view, object pointer, identity map or token. Both successful and failed paths free
all preparation allocations. The report is a copied observation at this call,
not reusable authority; any subsequent host or guest mutation invalidates it.

The caller supplies an initialized ordinary VmState owned by the current thread,
with module/root identity matching the borrowed module, no running collector,
active frames/reference regions/handlers and an empty operand stack. The caller
must consume a previous result before this check, just as the proposed invocation
contract requires. In combined builds private_record_dag must be false. This
standalone query does not certify decode/constants/callback/dispatch readiness;
those remain separate future invocation preflight obligations.

The output must not overlap module/state/owned heap or preparation storage. Only
one output exists; it is assigned once after all work, with its final copy charged
in advance. Every failure preserves output, complete caller state, source module,
refcounts, intern links, cycle buffer, colours, counters and payload bytes. No
retain/release, GC, intern lookup that mutates a cache, or state normalization is
allowed. Only this query's own ledger and allocations change.

## Managed-storage precondition

All supplied object pointers and backing buffers must be live managed storage
with valid readable allocation extents and balanced ownership edges, constructed
through the ordinary VM heap contract. The caller must not race or mutate them
while this thread checks them. As with existing ordinary C heap APIs, forging an
address or lying about allocation extent is outside the API precondition. Checked
size arithmetic and count/capacity checks cannot establish malloc extent from a
header. I do not advertise memory-safe inspection of arbitrary C objects.

Within that precondition I refuse inconsistent tags, counts, nominal fields,
required element kinds, cached hashes and table topology. Corruption tests keep
all dereferenced storage valid, and restore malformed metadata before destructor
calls where that metadata is itself a destructor precondition. A new allocation
registry or general hostile-pointer checker is not part of this proposal.

## Collector membership without mutation

First validate cycle_count<=cycle_capacity and coherent pointer/capacity storage,
checked capacity*sizeof(pointer), gc_running=false. Charge the complete buffer
scan before traversing entries. Existing constructor/notification/collection
logic gives a unique pointer per buffered entry: notification appends only after
observing !buffered and sets that flag before returning; collection moves the old
buffer aside. This is an invariant to verify with actual ordinary fixtures, not
an assumption that zero-ref entries are absent.

Each occupied buffer position must be a managed container of one of the ordinary
collector's child-bearing tags (ARRAY, STRUCT, UNION, TUPLE, HASHMAP, CLOSURE),
have buffered==1 and either positive refs/PURPLE or zero refs/BLACK. Duplicate
buffer entries refuse; they are not silently deduplicated. These GC-only objects
receive membership metadata, not a static origin obligation, and their payloads
are not recursively traversed merely because they are unrelated suspects.

For graph-reachable containers, require positive refcount. Accept BLACK or PURPLE
when unbuffered; PURPLE without buffering is the actual buffer-growth failure
state. Buffered reachable objects must match exactly one previously scanned
membership entry. GRAY/WHITE are trial-deletion states and refuse an idle check.
A graph object claiming buffered while absent from the buffer, or an unbuffered
object listed in it, refuses. Strings are not collector containers: require
positive refs, BLACK and !buffered; their byte identity remains independent from
static origins. Unrelated zero-ref buffered BLACK entries remain untouched.

A single charged identity table can hold GC-only and graph objects. A row uses
object pointer, u32 obligation (UINT32_MAX means no graph assignment), one DFS
colour and one membership flag, with reserved bytes zeroed: expected16 bytes.
Hashing a pointer is only runtime identity comparison, never provenance recovery.
Collector membership does not turn an unsupported object into an admitted global.

## Singleton historical obligations

Every allocated global slot through global_capacity must match its copied
historical tag set, regardless of global_count: LOAD_GLOBAL can read reserved
capacity. Slots above the copied global count must be VOID; only slots beyond
actual global_capacity are logically VOID and
must be allowed by their row. Unknown historical rows already refuse the module
proof. Direct-map globals remain unsupported. Scalar/string occurrences get no
static origin assignment. For ARRAY/STRUCT and nested HASHMAP occurrences, filter
only by exact runtime kind, then require one static origin before inspecting
shape. Multiple candidates do not become unique by picking a convenient shape.

The first graph occurrence assigns that singleton structural obligation to its
runtime identity. Every later occurrence must demand the same singleton; a
conflicting or ambiguous alias refuses. Different objects may share one site.
This is an externally checked obligation assignment, not recovered provenance.
The full historical/future field and child rows constrain every object, not only
current final children. Even a previously completed object must satisfy each
occurrence's additional nominal/exact depth/leaf descriptor before reuse.

For ordinary records I require `def_idx==origin.record_ordinal`, then query the
copied descriptor for that ordinal and require `descriptor.layout==origin.layout_index`.
I use the resulting layout for schema, nested nominal and field checks. Ordinary
STRUCT_NEW/STRUCT_LITERAL/AGG_RECORD store the bytecode ordinal directly; private
handlers can remap it. This query neither compares an ordinary def_idx directly
with a layout index nor relabels existing objects. A nonidentity ordinal/layout
fixture is required when permitted by the declaration grammar. Future public
handlers must preserve ordinary ordinal identity unless a separate compatibility
design is reviewed. Each copied descriptor query receives its four-probe charge.
Records require exact origin nominal identity, field count/storage and every
historical field summary, not pointer identity or field names. The initial boundary follows the existing
selected representation with field_names==NULL; support for externally attached
reflection names requires separate evidence. Arrays require exact packed/boxed
storage, length<=capacity, checked element width and every historical child
obligation, including depth1/2/3 and required_elements. Empty ARRAY children must
still obey declared depth; an empty payload does not erase nominal obligations.

A DFS back edge to a GRAY graph identity refuses a cycle without touching heap
collector colours. The table's DFS colour is unrelated to VmHeapHeader.colour.
Repeated BLACK identities are allowed only with the same singleton and all
occurrence descriptors. Static sites<=64 do not bound runtime object count.

Maps require the copied exact STRING→INT pair, valid count/capacity, zero tombstones
and exact filled/empty state bytes. Validate every occupied counted string key,
INT value, cached hash and probe reachability; duplicate equal-content keys refuse.
An equal-content separately allocated key is legal only as the unique stored key;
keys shared across different maps remain legal. Embedded NUL is valid counted data
here; I preserve the separate native target refusal. Missing GET remains INT|VOID
in recipes, but this query executes no GET. Map keys are validated inline, not
assigned static sites or pushed onto the DFS stack.

No retain occurs during validation, so UINT32_MAX alone is not an import failure.
A future dispatcher must preflight exact retain multiplicity before each mutation;
this query does not reserve or consume refcount headroom across calls.

## Numeric preparation ledger

My proposed automatic working struct holds the identity-table pointer (8 bytes),
u32 capacity/count (8), five u64 counters (bytes/work/graph/collector/edges:40),
and259 DFS frames (4144), for an expected4200-byte struct on this ABI. The copied
report is40 bytes. Copied query row scratch is charged explicitly as the sum of
`sizeof(NvmRecordHeapOrigin)`, `sizeof(NvmRecordValueOrigins)`,
`sizeof(NvmDeclarationLayout)`, `sizeof(NvmV2LayoutField)` and
`sizeof(NvmRecordArrayExecutionDescriptor)`; implementation must use those rows
repeatedly rather than recursively allocate hidden per-object scratch. Actual
sizes and allocation-oracle equality remain first-build qualifications, not a
measured claim. The fixed struct/report/row scratch is prepaid before use.

I start with the boundary plan's full reservation, never replace it with only
retained bytes or grant a fresh budget per object. Both limits remain existing
128MiB and33,554,432 logical work. Charge the fixed working struct/report/row scratch described above before traversal;
the259 frames are inside that struct and must not be double-charged. The259 path bound separates256 nominal
records+3 arrays from256 records+one map; keys/scalars are inline. An actual
constructible maximum must be measured, not inferred from the syntactic bound.

Identity buckets start16 and double before occupancy exceeds one half. Reserve
old+new bucket payload during growth with checked arithmetic; charge all new
zeroed bytes, old-capacity scan and two units before every probe/reinsert. GC-only
membership entries consume the same capacity and budget. No arbitrary64-object
limit or uncharged auxiliary set is introduced. After growth, retained storage
can be tracked separately but charged reservations are never refunded.

Charge32 for the idle/header checks, eight per global slot, eight per cycle-buffer
entry before inspection, and two per table probe. Charge64 before filtering static
origins, four per copied global getter and existing documented bounds for other
copied getters. Charge16 per physical header and eight per field/array element;
charge each counted string length+16 before byte/hash/terminator checks. The map
validator retains16 header, eight per bucket, length+1 hash, four per probe and
length+1 equal-length comparison (one for differing lengths), all against this
same ledger. Prepay final sizeof(report) before output assignment. Overflow, cap
exhaustion and failed table allocation return LIMIT or MEMORY without partial
output or caller mutation. These are preparation bounds, never guest fuel.

## Required qualification before any dispatch proposal

1. Four unchanged annotated originals and the separate accepted source-global
   products: empty initial state, repeated result consumption and initializer
   replacement, retained old alias and unchanged original authority refusal.
2. Actual ordinary buffered aliases, shared record/map/key identities, buffer
   growth allocation failure leaving unbuffered PURPLE, and unrelated deferred
   zero-ref BLACK. Retain byte snapshots of GC metadata before/after the query.
3. Same-singleton shared imports pass; two physical globals sharing an object
   with distinct or ambiguous static obligations refuse. Include later-write
   incompatible alias cases, not only identical current contents.
4. Reachable cycles refuse; subsequent original ordinary release/GC/destroy still
   reclaims them. No metadata repair or forced collection is part of the check.
5. Count/shape/topology corruptions use live in-bounds storage: duplicate keys,
   unreachable probes, wrong cached hash, tombstones, wrong leaf/nominal children,
   valid counted-NUL keys, missing/duplicate collector membership and bad idle
   states. Preserve full caller bytes, counters and output on every refusal.
6. Identity-table collisions/growth, maximum supported path, precise cap/+1 and
   overflow, every allocation prefix and independent retry. Compare independent
   allocation live/peak accounting; caller-owned payload is reported separately.
7. Old private observation/runtime/heap neighbors and six legacy budget rows remain
   exact. No VM/native/publisher selection or handler runs through this query.

## Source checkpoint charge precision

The draft state has the proposed4200-byte fixed payload,16-byte buckets/frames
and40-byte report (compile-time assertions and actual ordinary-query qualification). The
prepaid copied-row scratch conservatively includes three simultaneous origin
rows (walker, selected origin, filter candidate), two value-origin rows and two
NanoValues, plus layout/field/descriptor, boundary counts, origin counts and map
work-header copies. This is a logical payload reserve, not a claim about compiler
ABI stack-frame padding; helper depth is fixed and graph depth uses the259-frame
array. Actual sizeof values are qualified; the final receipt records allocation and old-budget comparisons.

Packed arrays have no boxed child summary in the existing proof. Their imported
physical values use the exact proved declared scalar kind after existing packed
conversion; boxed arrays retain the full historical tags/origins. No raw scalar
write set is misrepresented as the tag of an already converted stored value.
The ordinary map wrapper shares the complete charged topology loop with the
private validator through a static implementation parameter. No exported boolean
can skip validation or assert header trust; old private wrappers still select
the original !buffered header and preserve their prior schedule.

## Qualified implementation boundaries

I measured the ordinary owner layouts unchanged: VmState803992 and VmHeap6312.
The test harness keeps functions with these large automatic owners uninlined;
the initial O1 sanitizer binary reserved0x671160 bytes in main alone and overflowed
when calling another owner-sized snapshot. Provider optimization and sanitizers
remain unchanged. This is a harness frame correction, not a runtime stack change.

The global scan covers actual allocated global_capacity, including readable slots
above global_count. Missing allocated values are never synthesized as VOID. A
real LOAD_GLOBAL assertion reads17 from slot0 while count is0; a non-VOID value
outside the copied historical summary refuses. The earlier count-based draft was
corrected during source review before any build.

The static positive-only reservation helper initializes the full boundary reserve,
then adds checked byte/work reservations before scratch or graph traversal. Normal
module-taking queries pass0/0. Whitebox exact-cap tests preserve the same graph
and pointer/hash inputs; +1 refuses without output publication. UINT64_MAX refuses
before validator-table allocation, with an already-created boundary plan. I do
not claim the module-taking query is allocation-free on every LIMIT result.

I qualify a constructible63-frame record/array path with64 origins. The259-frame
scratch reserve remains a conservative maximum; I do not infer a258-frame
positive case from that bound. Ordinary record ordinal0 mapping to layout1 is
qualified with an interleaved union declaration, without relabeling the object.
