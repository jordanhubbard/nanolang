# Private map observation interface — design checkpoint

Owner: owned_calls, #1001. Base: aggregate48e1508ae, private invocation3ec90edcb. The bounded design is reviewed for implementation; a source checkpoint and tiny tests precede broad qualification. No public selector, native consumer, publisher, admission or runtime implementation changes are authorized by this document. Observation is not an execution token or a proof query.

## Identity decision for review

The older array observation copies an address into uint64 identity. I do not propose that for this new interface. A stable numeric registry would introduce allocations and an additional identity lifetime; hashing addresses would introduce collision or disclosure claims. Instead, the first bounded interface copies value metadata and offers an explicit two-path alias comparison. Pointer comparison occurs only internally; no address, pointer-derived token, borrowed NanoValue or trusted context escapes. A returned comparison describes two heap endpoints in one idle owner at the returned epoch. It cannot be replayed as authority, compared across owners, or used to assert anything after another invocation. This replaces the illustrative identity field in the earlier invocation design and needs explicit review before implementation.

## Exact proposed declarations

All declarations remain behind NANO_RECORD_MAP_PRIVATE_RUNTIME in record_map_runtime_private.h. Existing owner, heap and VM layouts stay unchanged. No persistent observation cache or allocation is added.

```c
typedef enum { VM_RM_RESULT, VM_RM_GLOBAL } VmRecordMapRoot;
typedef enum { VM_RM_OBS_OK, VM_RM_OBS_INVALID, VM_RM_OBS_LIMIT }
    VmRecordMapObservationStatus;
typedef struct {
    uint64_t epoch, scalar_bits;
    uint32_t length, layout;
    uint8_t tag, element_tag, key_tag, value_tag;
} VmRecordMapObservation;
typedef struct { uint64_t epoch; bool same; } VmRecordMapIdentityComparison;

VmRecordMapObservationStatus vm_record_map_private_observe(
    const VmRecordMapPrivate *, VmRecordMapRoot, uint32_t global,
    const uint32_t *path, uint16_t path_count,
    VmRmObservationWork *, VmRecordMapObservation *);
VmRecordMapObservationStatus vm_record_map_private_lookup(
    const VmRecordMapPrivate *, VmRecordMapRoot, uint32_t global,
    const uint32_t *path, uint16_t path_count,
    const void *key_bytes, uint32_t key_length,
    VmRmObservationWork *, VmRecordMapObservation *);
VmRecordMapObservationStatus vm_record_map_private_string(
    const VmRecordMapPrivate *, VmRecordMapRoot, uint32_t global,
    const uint32_t *path, uint16_t path_count,
    uint32_t start, uint32_t length, VmRmObservationWork *, void *bytes);
VmRecordMapObservationStatus vm_record_map_private_same_identity(
    const VmRecordMapPrivate *,
    VmRecordMapRoot left_root, uint32_t left_global,
    const uint32_t *left_path, uint16_t left_count,
    VmRecordMapRoot right_root, uint32_t right_global,
    const uint32_t *right_path, uint16_t right_count,
    VmRmObservationWork *, VmRecordMapIdentityComparison *);
```

Observation is initialized to zero locally; layout is UINT32_MAX unless STRUCT. INT/FLOAT scalar_bits use an exact memcpy of the value bits; BOOL/U8 are normalized actual scalar values, VOID bits remain zero. STRING length is counted bytes; ARRAY length/element tag and STRUCT field count/layout are copied. HASHMAP length/key/value tags are copied only after full charged topology validation. Lookup succeeds with copied INT or VOID; a missing key is OK/VOID, never INT0. No returned structure contains a heap edge. String copying does not append a terminator and handles empty slices and counted embedded NUL exactly.

Same-identity requires both endpoints be valid heap objects; scalars are INVALID rather than silently value-compared. Different heap tags imply false. A shared object reached through distinct roots/paths compares true. Retained prior-result/global alias mutation after a failed run is visible at the new epoch; this is identity preservation, not mutation rollback. Destroying the owner invalidates all descriptive observations. There is no meaningful cross-owner identity comparison.

## Preconditions, failure and storage

Require prepared idle owner, owner thread, no active frames/stack, valid current image ownership, and published result for RESULT. RESULT requires global=0; GLOBAL requires index<global_count. Reject unknown root values. Path length is at most258; nonzero lengths require readable path/key/output ranges. A zero-length key or copy may use NULL. Path traversal accepts only STRUCT field indices and ARRAY element indices. Maps and strings terminate paths; map keys are never interpreted as path indices.

Caller buffers must be valid, and output/work storage must not overlap each other, input path/key storage, the owner or its owned graph. This is an explicit private C API precondition, not a claim that arbitrary memory allocation extents can be discovered. Detect direct output/work/input range overlaps using checked uintptr_t arithmetic before any writes, including end overflow; graph-disjointness remains a caller ownership precondition. Two input paths may overlap each other since neither is mutated. Invalid arguments or malformed work header leave outputs unchanged. `used<=limit<=33554432` and a valid work header are required at entry. A malformed header is INVALID even if limited=true; an otherwise valid already-limited counter returns LIMIT without walking. Nonzero reserved/padding bytes are not protocol inputs.

All semantic outputs are staged locally and committed once on OK. On invalid graph or LIMIT, observation/copy/identity outputs stay byte-identical. Work is explicitly cumulative: charges already paid remain, and exhausted work sets limited=true without adding the failed charge. It is not transactional output, runtime guest fuel, or preparation accounting. The comparison query charges both endpoint traversals independently, even when paths are equal. Work storage cannot alias an output to make their failure rules ambiguous.

## Charged traversal and exact byte/probe schedule

No query allocates or retains. Do not reuse vm_rmx_value_valid's recursive array scan or any uncharged whole-map lookup. The new owning-TU observation walker must check only the visited shell and selected edge, with every scan separately charged. Counts and checked arithmetic precede loops. Closed constructor-owned, non-concurrently mutated storage is required; corrupt test fixtures must remain valid allocated objects and be restored before destruction.

1. Charge16 before owner/root validation and root read. Validate fixed-size idle identities without traversing guest structures.
2. Charge16 before validating each visited value shell, including the endpoint. Scalar tags are exact; STRING verifies header/refcount/buffered/data/length+terminator bounds; ARRAY verifies header/count/capacity/storage/unboxed consistency; STRUCT verifies copied nominal row, field count/storage/no reflective names. No child array or record subtree is scanned here.
3. For each path edge, charge8 before reading its index and selected child. STRUCT checks copied field tag and exact nominal/array-depth/map-pair descriptor for that edge. ARRAY checks its exact element/storage tag and selected child; packed reads use the existing representation conversion. A bounded expected-depth/leaf descriptor accompanies traversal through nested arrays, so empty depth-mismatched arrays cannot pass by lacking elements. No unvisited sibling is read or implicitly credited as independently validated.
4. At a HASHMAP endpoint, invoke existing vm_rm_heap_validate with the same work object:16 header,8 per bucket, key length+1 before hashing each occupied key,4 per probe, and length+1 for equal-length comparison (1 for differing length). This establishes occupied count, cached hash, exact STRING→INT types, unique keys and probing reachability. It is not replaced by a header-only check.
5. Lookup then charges key_length+1 before hashing the caller's counted bytes. A new private heap-TU charged byte-lookup helper takes `(const VmHashMap *, const void *, uint32_t, VmRmObservationWork *, NanoValue *)`; it has the explicit prerequisite of the just-completed validation in the same idle call, not an exported trust token. Each probe charges4 before bucket access; each comparison charges key_length+1 when lengths match, otherwise1 before comparing. At most bucket_count probes. It uses the exact existing FNV and probing arithmetic, preserves counted NUL, and never creates/interns a STRING. It returns only copied INT/VOID. Do not call uncharged fast vm_rm_heap_get after validation.
6. String copy validates checked start/length within counted length and charges length+1 and the final output commit charge before the single memcpy. Endpoint string metadata does not copy or hash all bytes unless explicitly requested. No per-byte guest loop is hidden behind a fixed charge.
7. Alias comparison charges1 before its final internal address equality, after both endpoint validations. Final output assignment uses a fixed1 precharge for each API. All charges use overflow-safe subtraction against the existing observation cap.

The schedule counts bounded logical work, not literal CPU instructions. Full map validation can be expensive for adversarial collisions but exhausts the existing cap before additional hash/probe/comparison work. No fixed uncharged capacity scan is allowed. Lookup does not revalidate topology a second time. No observed object is modified, including refcounts and cached hash.

## Qualification obligations before adoption

- Actual four retained original inputs through private execution, source/image independence and result/global observation. Scalar bit patterns, exact nominal rows, arrays depth1/2/3, shared/distinct map and record endpoints.
- Missing GET VOID versus INT0; empty/count-NUL/UTF-8 keys; equal-content distinct key allocations; collision-heavy tables; replacement and growth. Compare every lookup with real guest HAS/GET/LEN results.
- Shared aliases and prior result/global identity after failing ASSERT; show same-identity true without exposing pointer bits. Epoch advances on successful and failed actual invocations; busy/thread/wrong-root/path refusals.
- Measure the actual supported maximum valid path and retain259 refusal without inventing a path258 positive; empty wrong-depth arrays, malformed selected fields, map duplicate/unreachable bucket/hash/count corruption. Restore physical corruption before teardown.
- Exact charge total and one-below LIMIT for each operation, combined calls sharing work, UINT64MAX/overflow invalid headers, already-limited behavior. Output sentinels preserved; demonstrate no allocations/refcount mutation during successful and failed observations.
- Explicit output/work, output/path, output/key overlap controls plus invalid output range arithmetic; graph-disjointness documented, never tested by corrupting the live owner through an output pointer.
- Old array observation bytes/behavior and all legacy budget/owner sizes unchanged. Public/native/serializer/publisher refusal remains required. The parent approved bounded implementation after these refinements; this is not a new release acceptance claim.

Review refinement: the heap-TU byte-probe helper is static/internal. If cross-TU access is needed, the private wrapper performs full topology validation then the static probe in the same query; it exposes no trusted flag or reusable validation token. The VM walker must not redundantly validate the map before this combined wrapper. Other map observations retain the existing full validator.
