# Private STRING→INT map heap adapter and teardown checkpoint

Status: contract-only above preparation34d080e06 / aggregate96504adf5, #1001 open. I own this next scoped audit. No dispatch/public/native/publisher change is authorized by this document. DELETE, KEYS, VALUES and heap-valued map values remain outside the selected profile. The existing shared VM still implements its old legacy opcode behavior.

## Existing behavior and exact mutation sites

`src/nanovm/heap.c:846–967` creates a map shell and16 buckets, probes using cached STRING hash and counted equality, and doubles at occupied load greater than0.75. `hm_find_slot` probes at most bucket_count, tracks a tombstone candidate, and stops at an empty bucket. `hm_resize` allocates a replacement table, rehashes existing occupied entries without retain/release, then commits the replacement and frees the old table. It assumes unique reachable entries; ignored `found` during rehash is not an independent topology proof.

Existing `vm_hashmap_set` first looks up the key. Replacement retains the new value, changes the value, then releases the old value; it does not retain the supplied key or allocate. Insertion may resize, repeats the lookup, writes key/value/state, retains both operands and increments count. It borrows caller operands. Public `HM_SET` at vm.c4524 pops value/key/map, checks receiver and declared tags, calls this helper, releases key/value and returns the original map edge on success. Failure releases all three popped operands and reports the first type or memory failure. Public GET obtains a borrowed value, retains it, consumes map/key and publishes the value. Missing GET is tagged VOID, not INT0. HAS consumes map/key and LEN consumes map. These opcode handlers are read-only in this phase.

The new checked adapters share the exact hash/probe/resize primitives. They do not duplicate a hash table or infer presence from HAS. Their direct qualification does not count as actual selected VM dispatch qualification.

## Private interfaces and storage preconditions

A new private heap header exposes these functions only for the qualification build:

```
VmRmHeapStatus vm_rm_heap_new(VmHeap *, VmHashMap **out);
VmRmHeapStatus vm_rm_heap_set(VmHeap *, VmHashMap *, NanoValue key, NanoValue value);
VmRmHeapStatus vm_rm_heap_get(const VmHashMap *, NanoValue key, NanoValue *out);
VmRmHeapStatus vm_rm_heap_has(const VmHashMap *, NanoValue key, bool *out);
VmRmHeapStatus vm_rm_heap_len(const VmHashMap *, uint32_t *out);
VmRmHeapStatus vm_rm_heap_validate(const VmHashMap *, VmRmObservationWork *);
```

Statuses are OK, INVALID, MEMORY, RETAIN_OVERFLOW and LIMIT. LIMIT belongs only to explicit observation validation. There is no per-guest-operation work/fuel cap. Scalar/result outputs remain unchanged on failure. Operands are borrowed; adapters neither consume caller references nor publish an extra map reference. Output/work memory must not overlap map/string storage. The heap and constructor-owned allocations must be live; no API claims to validate arbitrary pointer allocation extents.

Fast operations require the closed-construction invariant: fresh maps come from checked NEW; only checked SET mutates them; no host imports, callbacks, concurrent mutation or DELETE. A future invocation context must establish this before reaching these helpers. No hidden heap registry, new static-origin pointer mapping or public trusted token is introduced here. Header checks enforce map object tag, positive refcount, exact STRING/INT tags, valid nonzero bucket storage, count within capacity and zero tombstones. Keys require a live counted STRING header. The full validator additionally checks every state/count/key/value, key hash versus counted bytes, and **exact bucket** returned by lookup for each occupied key; duplicate equal keys, unreachable entries and incorrect cached hashes refuse. Cached-hash fast operations rely on immutability of constructor-owned STRINGs. The existing observation logical schedule and nonallocation LIMIT semantics remain those in NANOISA_MAP_RUNTIME_INTEGRATION_DESIGN.md.

`hm_find_slot` may become const-correct internally (it only reads map storage), with no public signature change. The full validator may use a charged equivalent probe walk to account each bucket/byte before inspecting it; it must compare its result to the shared unbudgeted lookup in qualification. It must not call an uncharged whole-table helper and retroactively claim bounded work.

## Retain-before-commit and failure order

NEW borrows a nonnull heap and publishes only after shell+bucket allocation success. Failed bucket allocation frees the unpublished shell; no object/stat publication occurs.

SET validates tags/header before lookup. On replacement it requires the existing value to be INT, assigns the new INT and returns OK: no key retain, no growth and no value heap edge. A key with refcount UINT32_MAX may therefore be used for replacement without a spurious retain-overflow failure. Counts and key identity remain unchanged, including equal bytes supplied via a different live STRING object in a defensive fixture.

On insertion, check the supplied STRING's **one** additional reference for headroom before any resize. If growth is required, call shared resize and recompute the slot; no saved pointer/index is reused after growth. After successful allocation/probing there are no remaining fallible operations: retain the key first, then publish key/value/state and count. INT values have no retain/release. This is a new private commit path; the old public helper's ordering is not silently changed. Both paths share the existing lookup/resize algorithms.

For valid closed storage, a fresh empty slot exists after successful growth; a rehash result unexpectedly reporting found or no slot is an invariant violation, not an allocator error. It must not be relabeled MEMORY or claimed to roll back a committed capacity change. Corrupted physical maps belong to validator refusal tests, not fast-operation inputs. Normal allocation or headroom failures before commit preserve original contents, bucket identity/capacity, refcounts and input ownership. The future VM handler, not the adapter, consumes popped operands once on failure; no handler is changed now.

## Narrow allocation statistics correction

The retained original probe (`evidence/map-runtime-accounting-20261021`) grows16→128 buckets and observes allocated2646/freed7126/objects0. Difference4480 equals growth capacity bytes. This is a stats defect; zero objects alone did not establish allocation freedom.

Proposed shared correction is limited to `hm_resize`: add an internal VmHeap argument, pass it from its sole existing caller in vm_hashmap_set and the new checked adapter, and after successful replacement commit add `(new_count-old_count)*sizeof(VmHMEntry)` to `heap->stats.allocated` when heap is nonnull. Every actual call here is growth by doubling, with checked capacity and multiplication. Preserve NULL-heap behavior of the old helper where it was otherwise meaningful; selected adapters require nonnull heap. Do not introduce a shrink operation or change probing, capacity thresholds, value semantics or error status. Failed allocation/rehash adds nothing.

This follows `array_grow`'s existing net-capacity accounting at heap.c619. `allocation_calls` remains the count of published object constructions, not every bucket allocation; num_objects and freed remain unchanged by resize. On final map release, current shell+capacity bytes are counted once as before. Stats are existing size_t cumulative counters, not a new overflow/fuel policy; the qualified finite fixtures stay representable. The independent allocator oracle separately sees old+new tables simultaneously during growth, counts temporary failed work and returns to zero after destruction. Stats do **not** claim allocator metadata, intern bucket storage, cycle-buffer storage or transient peak. Intern-table accounting is not changed.

Expected corrected original probe is allocated7126/freed7126/objects0. Failed growth preserves both counters; retry contributes only the successful capacity delta. Old INT→STRING maps retain replacement-before-release behavior and now receive the same truthful capacity accounting. Existing cycle collector eventually calls ordinary vm_release on map objects, so the same current-capacity release remains applicable; no collector algorithm edit is proposed.

## Private iterative teardown without a larger release stack

Current `vm_private_dag_release` supports only STRUCT/ARRAY parents and inline STRINGs, with259 stack entries. A dead HASHMAP currently reaches an assertion. For exact STRING→INT maps, add a private-map-gated inline case: visit occupied buckets, decrement/release each stored STRING edge using the existing inline STRING rules (including intern unlink and buffered/refcount checks), ignore scalar INT ownership, then free table/shell and account current-capacity bytes. Do not call vm_release recursively while the shared iterative stack is active. A live shared map decrements once and leaves its keys untouched until its last edge dies. Shared keys across maps are decremented once per actual stored edge.

**Layout/configuration proposal for review:** require the new private heap adapter qualification macro together with existing NANO_RECORD_ARRAY_PRIVATE_RUNTIME. Reuse its existing private_record_dag flag and259-entry storage; no new VmHeap/VmState field or enlarged old stack. Compile every VM-layout-dependent provider consistently in this private test configuration. Map preparation's previously measured macro-only812256-byte owner remains its original configuration; any later combined-macro owner size must be measured separately, never called identical by assumption. Old ordinary builds and old array-only builds do not compile the map release case. Old array physical admission still refuses map fields; combined-macro teardown is tested directly with exact constructed maps, not used to enable VM execution.

Map inline teardown adds no parent nesting:256 record parents can release a terminal map and its STRING keys without a257th parent frame. Separately, the existing256-record-plus3-array primitive-leaf maximum stays259. This does not admit map elements in arrays or invent a record→array→map source profile. Record→map and shared-key maximum chains need actual proof; no claim follows merely from this argument. Reflective names, heap-valued map entries, tombstones and buffered cyclic maps remain excluded from this private DAG case. Ordinary cyclic maps retain their existing release path.

## Qualification before any VM routing

1. Direct new/get/has/len/set with present/missing INT0 versus VOID; zero-length and counted-NUL keys; equal-prefix keys differing after NUL; collisions; repeated interned keys across separate maps; alias mutation. Compare counted bytes and exact identity/refcounts.
2. At a growth threshold, replacement succeeds with all allocations denied and with key refcount at UINT32_MAX. Insertion at maximum refcount refuses before allocating; failed growth preserves bucket pointer, counts, prior key/value entries, caller refcounts and stats. Reset only test-forged headroom after inspecting refusal, then independent retry succeeds.
3. Every direct NEW/growth allocation position, transient and persistent, uses independent allocation interception. Failure preserves sentinel/prior map and exact contents; retry and destruction balance live bytes. Corrected legacy growth probe plus independent transient peak accounting is mandatory, not an adjusted expected counter masking leaks.
4. Physical validation rejects duplicate keys, unreachable occupied bucket, malformed state/count/tombstone, wrong pair/value/header/cached hash. Output/work nonoverlap remains an explicit caller precondition, not a claim that arbitrary aliased or forged memory can be safely validated. Restore each deliberate corruption before teardown. Exact logical bound succeeds; one-below and arithmetic overflow preserve outputs. No malformed map reaches fast mutation by design.
5. Private iterative release with shared record/map aliases, shared interned keys, maximum record/array chain and partial construction cleanup; no recursion/collector allocation, no stack enlargement. Old array-only and combined private layouts both receive sizeof and old-budget comparison. New heap adapter tests never execute a selected module.
6. Existing test_heap_allocation_failure (especially old INT→STRING replacement/growth), ordinary map VM tests, cycle collection neighbors, private array teardown/allocation suite and constructor/image neighbors remain required. Retain original failure terminals. Strict Clang/GCC and sanitizer builds use consistent private macros. No full frontend/bootstrap is needed.

Planned source scope after review: heap.c plus a new private heap adapter header/include; test harness/driver/Make dependency wiring. No vm.c opcode/dispatch edits, no public verifier/load/serialization/native/publisher route, and no shared VmState/VmHeap declaration additions. The stats change is kept identifiable from private adapter and teardown changes in the diff and receipt.
