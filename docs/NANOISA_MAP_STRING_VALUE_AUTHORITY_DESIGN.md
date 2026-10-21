# Exact STRING-valued map authority: design checkpoint

I base this proposal on aggregate `200c4f850` and issue #1001. This is a design-only checkpoint. I have not changed admission, built providers, or rerun the proposed baseline. My completed STRING→INT DELETE component remains separate and frozen.

## Evidence, scope and baseline

My exact source is unchanged `tests/unit/test_map_field_operations.nano`: `Holder { values: HashMap<string,string> }`, SET, HAS, GET comparison with `"kept"`, LEN, DELETE, absent HAS, and both original shadows. The actual two-method operation gate in [the DELETE publication receipt](evidence/map-delete-publication-20261022/README.md) executes the STRING-valued case through current C and fresh C-hosted Nano products and explicitly expects absent authority. That establishes the source gap; it is not selected STRING-valued qualification.

Before implementation I require a separately retained four-original baseline: C ordinary, current C-hosted Nano ordinary, independently selected C original shadow, and raw Nano original shadow. I preserve source bytes/assertions, actual producer/consumer hashes, raw verification/VM and O0/O2 native outcomes, current authority omission, and independently annotated selected refusal with prior-output preservation. Checked captured shadows and raw parser shadows retain distinct provenance. A generic original failure stops the phase. Installed stages are not current-source evidence; fresh bootstrap/platform acceptance remains independent.

I propose exactly one pair per selected module: existing STRING→INT or new STRING→STRING. Every map origin and field must match that pair. I do not admit a module containing both pairs, other pairs, HM_KEYS/VALUES, standalone map signatures/globals, or new STRING-producing/native domains. Mixed unsupported candidates omit all optional buffers atomically. Existing mixed map/array modules remain within their independently proved array contract.

## Representation and proof

Existing kind5/revision1 grammar already represents this pair. New STRING→STRING metadata has one sorted scalar row STRING, one pair referencing row0 for both endpoints, and existing exact nominal field bindings. Payload size is `28 + 12*bindings`; old STRING→INT remains `36 + 12*bindings` with sorted INT/STRING rows and pair(1,0). Kind/order/reserved/count rules, simultaneous-extension limit and schema are unchanged.

`record_map_analysis.inc` currently requires one STRING→INT pair, exact STRING keys/INT SET values, and historical scalar summaries. `managed_array_shapes.c` checks all encoded HM_NEW sites, including unreachable instructions, and emits INT|VOID for GET. I would derive the uniform value tag from the existing declaration read, require every creation/SET to match, and preserve keys/values history through deletion, branches, calls, globals and detached aliases. Strings remain abstract leaves without static container origins. The shared64-origin namespace does not grow.

GET becomes the exact declared value tag OR VOID; HAS does not establish presence. No key spelling, constant contents or later DELETE removes a historical write obligation. A wrong historical value followed by removal still refuses. Frozen origin rows retain their existing `(ordinal,pair)` representation and pair0 is justified only by the single exact pair rule.

Existing immutable recipe key/value/result masks and prepared field key/value bytes carry the new pair without extra persistent fields. Proof, recipe construction, copied-image independent reproof, both preparation paths, preflight correspondence and target classification must all agree. The admitted opcode set remains100; all256 opcode decisions remain checked. No partial executable enrollment is allowed.

## Phase1: checked heap ownership only

I first propose fixed-pair internal heap adapters, direct tests, typed physical validators and private inline teardown. Proof, recipes, image admission, public/private dispatch, native target and publishers remain unchanged in this first source slice. Existing INT adapter signatures/bodies and their zero-tombstone/tombstone work schedules stay intact. These functions are physical helpers, never an execution token.

Proposed signatures in `src/nanovm/record_map_heap_private.h` (names explicitly identify STRING→STRING):

```c
/* Under the existing private heap macro. */
VmRmHeapStatus vm_rm_heap_ss_new(VmHeap *, VmHashMap **out);
VmRmHeapStatus vm_rm_heap_ss_set(VmHeap *, VmHashMap *, NanoValue key, NanoValue value);
VmRmHeapStatus vm_rm_heap_ss_get_owned(VmHeap *, const VmHashMap *, NanoValue key, NanoValue *out);
VmRmHeapStatus vm_rm_heap_ss_delete(VmHeap *, VmHashMap *, NanoValue key);
VmRmHeapStatus vm_rm_heap_ss_has(const VmHashMap *, NanoValue key, bool *out);
VmRmHeapStatus vm_rm_heap_ss_len(const VmHashMap *, uint32_t *out);
VmRmHeapStatus vm_rm_heap_ss_validate(const VmHashMap *, VmRmObservationWork *);

/* Separate ordinary collector policy; no private DAG activation. */
VmRmHeapStatus vm_rm_ordinary_ss_new(VmHeap *, VmHashMap **out);
VmRmHeapStatus vm_rm_ordinary_ss_set(VmHeap *, VmHashMap *, NanoValue key, NanoValue value);
VmRmHeapStatus vm_rm_ordinary_ss_get_owned(VmHeap *, const VmHashMap *, NanoValue key, NanoValue *out);
VmRmHeapStatus vm_rm_ordinary_ss_delete(VmHeap *, VmHashMap *, NanoValue key);
VmRmHeapStatus vm_rm_ordinary_ss_has(const VmHeap *, const VmHashMap *, NanoValue key, bool *out);
VmRmHeapStatus vm_rm_ordinary_ss_len(const VmHeap *, const VmHashMap *, uint32_t *out);
VmRmHeapStatus vm_rm_heap_ss_import_validate(const VmHashMap *, VmRmObservationWork *);
```

I reuse existing typed statuses: OK, INVALID, MEMORY, RETAIN_OVERFLOW, LIMIT. No generic boolean trust/header-skip parameter is exported. Receiver/key/value inputs are borrowed from caller-owned live roots in the same heap. The caller establishes complete topology and collector membership in this invocation, or closed construction; managed-pointer validity, immutable string storage and exclusive access remain C preconditions. No callback/concurrent mutation is permitted. Output/work storage is disjoint from all inputs, graph/storage and each other. Every failure preserves output; successful output overwrites only caller-managed destination storage.

NEW returns one owned receiver edge or unchanged output on failure. GET returns VOID for absence; present GET returns one **owned** STRING edge after checked headroom and retain, before the caller releases receiver/key. There is no selected STRING API with an ambiguously borrowed return. The retained string is the VM's actual stored object, not a text copy. An output at saturated refcount refuses before mutation. Last-map-owner, key==stored-value and query-key==stored-key cases are mandatory tests.

SET replacement acquires one value edge before committing/releasing the previous value; it acquires no key edge. Equal stored/new value still requires the documented retain-before-release headroom. Insertion acquires one key and one value edge; if they are the same VmString, it requires two cumulative slots (`ref_count <= UINT32_MAX-2`). Validate all headroom before growth. After successful growth, retain both edges before publishing the bucket; no fallible step follows those retains. A failed growth preserves the old table, stored ownership and earlier committed deletions. Generic heap setters are unchanged unless a separately established defect requires review.

DELETE completes lookup, saves both stored edges, clears key/value to VOID and commits the tombstone/counts, then releases exactly two stored edges. It never reads either after release. If key==value, two distinct table-owned edges still retire exactly twice. The caller's query-key root remains until handler cleanup. Absent deletion remains allocation-free/no-op, including a validated fully live table or all-tombstone probe chain. INT deletion semantics remain unchanged.

Full validation checks both string objects' headers, exact tags, positive refcounts, counted lengths/hash/terminal byte and appropriate intern/collector invariants before dereferencing bytes. Equal strings need not share a pointer. Key probe topology, duplicate keys, bucket reachability, count/tombstone arithmetic and VOID tombstones retain the current rules. Ordinary buffered PURPLE containers remain buffered; strings remain BLACK/unbuffered. I do not normalize collector metadata or run GC to make validation pass. A typed import validator proves physical shape only; it does not confer authority.

Private teardown clears each live bucket and retires key and value inline, including equal identity and cross-entry sharing. Strings have no container children, so no extra release-stack frame is required. Each edge updates existing release/object/byte accounting correctly and intern unlink occurs only when the last edge reaches zero. Ordinary teardown remains cycle-aware and uses its existing collector. Phase1 direct fixtures can exercise this physical storage without authorizing any module.

## Later consumer boundaries

I would route both private and borrowed-public executors to the owned GET API only after exact recipe preflight. Public operands remain roots until all fallible adapter work succeeds, then consumed edges are released and the owned result is transferred to reserved stack storage. Private cleanup must likewise transfer the owned result before releasing map/key, preserving first error and prior result identity. Source SET/DELETE still produce the same receiver at ISA level and POP to source VOID.

Import must compare actual key/value types with the copied origin's pair, preserve singleton alias obligations and capacity-based globals, and validate every occupied string value. No runtime pointer becomes a static origin. Observation validates map topology and selected STRING shell, charges bytes/output before copying, and preserves output/work nonalias rules. Read-only internal lookup may borrow only within that same validated query; it cannot become a public retained-value API or reusable trust flag. Copied output remains pointer-free and epoch-scoped.

No persistent owner, Analysis, VmState or VmHeap field is proposed. Value tag is threaded through local parameters or existing copied fields; existing mutable analysis storage is not passed to execution. Old INT paths must retain exact allocation capacities, object sizes, six legacy budget rows, old map budgets and byte outputs. Additional string validation is dispatched only for STRING-valued maps and charged before scans. Deriving recipe tags must use already-performed reads or an explicitly reviewed schedule; I do not hide extra getters behind old work counters. Bounds/caps remain unchanged; actual new costs and retained/peak payload are measured before any fit claim.

## Native STRING|VOID and lifetime

Generic native `nmap_set` already copies a STRING value before releasing replacement and separately copies insertion key/value. `nmap_get` returns an owned text copy or `(nmap_value){0}`. `nmap_owned_get` registers only kind5 results and releases that text if owner allocation fails. `nvm2c_map_roots.inc` marks/sweeps returned-string owners using actual reachable roots. DELETE/destroy free both copied edges. These source observations do not qualify the selected canonical route.

Native simulation in `nvm2c.c` always pushes `NVM2C_VK_VALUE` for HM_GET; its optional child shape is tied to the map value shape. Emission uses `stack_push_value(nmap_owned_get(...))`, preserving absent kind0 and present kind5. TYPE_CHECK on this carrier compares `v.kind` with the requested tag. A demanded string extraction emits `nvalue_require_string`, which aborts unless kind5 before returning text. I require actual selected missing GET→TYPE_CHECK/branch tests and an exact wrong-use abort test; no fabricated empty string, unchecked NULL dereference, or inference from HAS is accepted.

`private_map_shape` currently fixes the value child to NVM_SHAPE_INT, and `private_map_pack_field` rejects a non-INT copied pair. Both are explicit later gates alongside `record_map_native_target.inc` correspondence. They must derive the independently copied uniform pair, preserve exact nominal field identity, and keep old INT emitted bytes unchanged. Native eligibility alone must not authorize execution or publication.

I retain the existing NUL-free executable PUSH_STR/STR_CONCAT closure and all other STRING-producer, STRUCT and nonidentity-ordinal restrictions. STRING GET extends value flow only from tables whose complete SET history has that same closure. VM counted-NUL behavior remains independent. Native acceptance requires constant/concat values through locals, helper calls and tails; GET retained across deletion/map destruction; absent GET; replacement; forced GC; O0/O2 ASan/UBSan; compiler allocation prefixes; and generated-process exact allocation-abort plus independent retry. No general counted-native string redesign is proposed.

## Ordered qualification and publication

1. Retain the four unchanged original baselines and independent annotation refusals before admission.
2. Review and qualify phase1 heap only: ownership/refcount saturation, equal-content distinct strings, same-object key/value, last-owner GET, collision/tombstone chains, growth OOM, corrupt-state restoration, ordinary buffered aliases/GC, exact work limits and all allocation prefixes. Compare old INT bytes/work/allocations and struct sizes.
3. Review one atomic consumer source checkpoint across proof/recipes/image/preparation/import/observation/both dispatches/native shapes and target. Preserve all-history negative controls, independent copied corruption refusal and output atomicity. Run actual public and private paths; no partially enabled selector.
4. Qualify unchanged original modules, independent shadows and focused lifetime/missing-result fixtures. Retain failures before correcting scope or assertions.
5. Only then extend exact C TypeInfo and Nano declaration-bound wire generation. Preserve raw-vs-checked provenance, independent ordinary/shadow queries, full staged VM/native intersection and INVALID/MEMORY hard-error sentinels. Uniform STRING→STRING publishes; mixed distinct pairs or unsupported operations omit the entire optional candidate. Do not syntax-filter metadata or assert validity from UNRESOLVED.
6. Run actual paired source/public transport/VM/native, allocation denial/recovery, old byte/budget gates and later fresh-stage/platform qualification. Other map values, key types and wider #1001 profiles remain separate.

I propose first-patch ownership only in `src/nanovm/record_map_heap.inc`, its internal header, the exact private teardown branch in `heap.c`, and direct heap tests. Later edits would touch map analysis/frozen/recipe/target code, private/public preparation/context/import/observation, native shape helpers, and finally both publishers. No first-phase opcode, proof, shared dispatch or publisher enrollment is authorized by this document.

## Baseline and proposed first source patch

The [four-original baseline](evidence/map-string-value-baseline-20261022/README.md) now passes original verification/VM and eight sanitized native executions. All four exact annotations are independently DESCRIBED by declaration getters, then UNRESOLVED at the current STRING→INT proof boundary. This does not authorize execution of those annotations.

For source review I propose the following heap-only patch, still unimplemented:

- Add the fixed `_ss_` interfaces above and new exact-pair private/ordinary header helpers in `record_map_heap.inc`. Keep old INT wrappers, validator bodies and schedules unchanged. Internal helpers may share new SS logic but no caller-provided bypass/trust flag.
- Add a new SS full validator with the existing16 header,8 per bucket,4 per bounded probe and counted-key comparison schedule. For each occupied value, additionally precharge16 logical header work, then checked `length+1` before hash/terminal-byte traversal. Charge repeated occurrences even when pointers alias; no hidden de-duplication table or allocation. Preserve the current tombstone scan/VOID/count/reachability policy. Exact caps/refusal/recovery will be measured against actual work.
- Intern-list ownership remains a managed-heap C precondition. Existing import/string helpers validate immutable header, counted hash/terminal byte and collector colour; they do not discover arbitrary allocation extent or prove intern-list membership. I will not call those checks a registry. Tests cover actual interned strings plus safely constructed distinct equal-content managed strings, and teardown unlinks only the actual last owned edge.
- Add checked insertion two-edge/replacement one-edge headroom and commit ordering; GET acquires one checked owned edge; DELETE clears both edges before release. Reads/presence/length never retain. All GET failures preserve out, and missing GET writes VOID successfully. No opcode handlers call these adapters yet.
- Add a separate exact STRING-value branch to private map teardown. Keep the INT branch text/behaviour intact; clear saved edges and retire both strings inline without recursive container frames. Ordinary generic teardown/statistics are untouched.
- Add direct heap tests in a dedicated test harness using actual ordinary/private heap providers: last-map-owner GET, saturated retained result, key==value insertion at MAX-2/MAX-1, replacement same identity/max key, distinct equal-content replacement, aliases across buckets, all-tombstone/full-live absent delete, growth OOM after deletion, counted-NUL strings, corruption/restoration, allocator prefixes, exact work limits and zero remaining owned allocations. Keep old INT direct gates and numeric size/budget oracles.

No Analysis/Value/VM/heap fields, proof getters, recipe masks, preparation, import selector, observation API, dispatch or native/publisher admission change in this first patch. Physical validators are tested directly; later module-taking import dispatch remains closed until consumer review.

I preserve the distinction for validated completely full live tables: new SS GET/HAS return VOID/false after bounded lookup and DELETE is an absent-key no-op. SS SET retains a separate no-slot refusal boundary; I do not claim full-table insertion support. Physical header validation presupposes genuine managed storage, correct owned-edge reference counts and intern ownership; it cannot prove allocation extent or repair forged ownership counters.

## Second-phase consumer interface checkpoint (review only)

I freeze physical heap commit `a33d1025a99b0740e3324e62f8c66983ffebb3db`. The following is a source-only audit and proposed atomic consumer change, not implementation approval. I do not change publishers in this phase.

### Uniform pair without stored fields

I keep `NvmRecordMapAnalysisContext`, `Analysis`, frozen facts, execution/image owners, both runtime preparation owners, invocation views and `Nvm2cBuf` unchanged in size. In `record_map_analysis.inc`, `map_declarations_exact` returns the single exact value tag through an output argument only after its existing charged counts/pair/two-scalar queries succeed. `analyze_boundary` keeps that byte on its C stack and threads it through `walk_map`, `map_operation`, `final_check`/`map_final` and all-code HM_NEW checking. Old wrappers pass the unchanged default and preserve NULL-map behavior. No getter runs per proof opcode. `maps->keys/values` remain historical scalar summaries; STRING values add no static container origins.

For recipes I add an internal successful-analysis out-tag alongside the existing frozen-facts/stacks output, retaining the old wrapper signatures. The output is published only with complete successful proof; every failed proof preserves it. `rme_prepare_boundary` receives this local tag and calls `rme_tags(row,value_tag)` during its existing charged instruction copy. This avoids another analysis or declaration lookup. GET receives `BIT(value_tag)|BIT(VOID)`; HAS stays BOOL, LEN stays INT and NEW/SET/DELETE stay HASHMAP. Pair count remains exactly one, and declaration key remains STRING. No opcode is added: all 256 cases and the current 100-supported count remain exact.

Both preparation layout passes already visit/query every map field under their existing 40-work-per-field schedule. They return a local uniform value tag (requiring at least one bound map field and exact agreement), then feed it into their existing coverage/correspondence passes. The prepared 20-byte field rows retain their existing key/value bytes, and 80-byte instruction rows retain existing tag masks. There is no owner field or repeated pair lookup. Copied-image reproof independently reconstructs these rows; it cannot trust a source row or caller tag.

### Exact affected sites

| File/site | Current INT boundary and proposed narrow change |
| --- | --- |
| `record_map_analysis.inc` `map_declarations_exact`, `map_operation`, `map_final` | Single pair policy, SET argument and complete historical value masks use the checked local tag. |
| `managed_array_shapes.c` `walk_map` HM_GET and all-code HM_NEW scan | GET keeps optional VOID; every encoded NEW, including unreachable instructions, must match the exact local pair. |
| `managed_record_map_execution.inc` `rme_tags` and successful analysis call | Parameterized value/result masks, same obligations and allocation/work schedule. |
| `record_map_vm_prepare.inc` and `record_map_public_prepare.inc` layout/coverage passes | Preserve field-pair getters and their charges; local tag controls exact copied recipe/operand checks. The supported count stays 100. |
| `record_map_vm_context.inc` and `record_map_borrowed_context.inc` value shells, field checks, recipe preflight, HM_NEW/SET checks and map executor | Select fixed INT or SS physical functions only after copied-pair/actual-header correspondence; owned SS GET transfers one output edge. |
| `record_map_import.inc` map-origin branch and physical validation | Retain the already-charged origin getter's key/value locals, compare actual header pair, then call exact INT or SS validator; no second origin query or trust flag. |
| `record_map_vm_observe.inc` map shell, full walk validation, lookup and copied result | Dispatch by validated exact field/header pair, include SS value byte work, preserve copied output atomicity. |
| `record_map_native_target.inc` recipe correspondence and internal successful target output | Exact value/result masks, same independently copied image and unsupported-domain checks. |
| `nvm2c.c` `private_map_shape`, `private_map_pack_field`, classifier chain | Constrain every receiver possibility to the single copied pair; do not merely accept either scalar shape. |
| `nvm2c_map_private.inc` target/image/core chain | Thread only the successful target's stack-local value tag; absent map route uses VOID and ignores it. |

The numerous INT root-signature, LEN, index, numeric and accepted packed-U8 conversion checks are not pair sites and remain unchanged. `record_map_frozen.inc` pair-zero origin rows remain justified by exactly one declaration pair; their 8-byte rows and existing getters remain unchanged. `record_map_admission.inc` zero-arity INT entry and VOID/INT initializer rules remain unchanged. Old INT heap bodies remain frozen; I do not generalize them into new semantics.

### Preflight and transfer

Both preflights first validate exact copied recipe masks against the independent invocation-local tag threaded as specified below; they never derive the expected tag from the same instruction row under examination. NEW compares both encoded operands; all other map operations compare the actual receiver pair with the recipe. SET checks exact STRING or INT value before mutation. SS value/string header and cumulative headroom checks remain in the reviewed fixed adapter. GET invokes owned SS GET before releasing receiver/key; the resulting edge is transferred into already-reserved stack storage without another retain. The old INT borrowed scalar route stays intact. Private popped operands remain owned C locals until cleanup; ordinary operands remain stack roots until adapter success. Saturation failure, same-key/value aliases, last-map-owner result, first-error cleanup and ordinary GC buffering need actual both-dispatch controls. NEW can take the preflighted encoded value operand as an explicit handler argument; no invocation field or repeated getter is needed.

I preserve closed-transition induction for shallow map/array shells. Full physical SS validation belongs at fresh import and explicit observation, not on every instruction. Import retains singleton structural assignment, global-capacity scanning and ordinary record ordinal/layout mapping. Strings remain inline scalar leaves, so the identity table and 259-frame bound do not grow. SS validation charges every stored value occurrence before byte/hash/terminal work, including duplicates of the key or another value. Old INT imports choose their exact old validator and schedule.

### Observation copy interface

Existing `vm_record_map_private_lookup` can already copy tag and length through `VmRecordMapObservation`, but it cannot copy a stored STRING's bytes: field paths do not traverse map keys. I propose a private observation-only `vm_record_map_private_lookup_string(owner,root,global,path,count,key,key_length,start,length,work,out)` alongside the existing string-copy API. It performs same-query full selected-map validation and the existing bounded byte probe, keeps its borrowed result entirely inside that call, checks STRING/range, prepays `length+1` plus final commit, then copies bytes. It retains nothing and exposes no pointers/identity token. Missing lookup returns the existing copied VOID observation through lookup; the string-copy operation refuses absent/non-STRING without touching output. Output/work/path/key/root storage remain disjoint C preconditions plus existing bounded argument checks. Exact-cap/refusal, counted-NUL output, thread/epoch/idle state and prior-output controls are required. A static internal SS observe-lookup wrapper must validate SS topology before using the shared static byte probe; no reusable trust flag is exported.

### Native local tag and work

I propose an internal `nvm_record_map_native_target_pair(image,allowance,value_tag_out)` with the existing `nvm_record_map_native_target(image,allowance)` wrapper forwarding NULL. Only ELIGIBLE publishes the tag, after the full scan; UNRESOLVED/INVALID/LIMIT preserve it, and the emitter never reads it on failure. The exact getter chain is declaration counts, pair zero, key scalar and value scalar: four bounded logical queries, no allocation, one pair only and outputs disjoint. I include those four within the existing 64-unit header precharge: four getter probes, twelve image/plan/count access and identity guards, sixteen entry/count guards, sixteen initializer guards and sixteen pair-policy/output bookkeeping units. This is bounded logical work, not a machine-instruction or allocator-time claim. Per-record 32, per-function 16, per-instruction 128+decoded-width and executable literal byte charges stay unchanged. I will qualify exact old target-work outputs rather than infer them from the formula.

The concrete native chain is `nvm2c_map_image_emit` → internal target helper → `nvm2c_emit_core(...,map_plan,value_tag)` → both `classify_function` call sites (currently lines 7896/7923) → `classify_function_body` → `private_map_shape` and `private_map_pack_field`. All use a stack argument; the non-map core call passes VOID. The packing helper also compares its existing exact field query result with this tag. No repeated uniform-pair getter occurs during shape convergence. The emitter has no separate logical work counter matching target validation, so I do not charge emission retrospectively to that target allowance or invent one. Existing bounded target/query accounting remains separate from actual C translation.

Generic native simulation already represents HM_GET as `NVM2C_VK_VALUE` with optional child shape. Its runtime `nmap_owned_get` returns kind 0 on absence and kind 5 on a stored STRING; TYPE_CHECK tests the kind, and `nvalue_require_string` must abort on kind 0 before dereferencing. I retain this distinction through the new exact shape constraint. Actual missing-key branch success, exact absent-value string-use abort, forced-GC helper/local/tail/result ownership and native allocation failures are mandatory. The existing NUL-free literal/STR_CONCAT intersection, STRUCT exclusion and ordinal/layout restriction remain; I do not extend the general string-producing domain.

### Accounting and atomic enrollment gate

I preserve Analysis 39,840, the opt-in analysis allocation size, 8-byte frozen map-origin rows, 20-byte runtime fields, 80-byte copied instructions, 14,512-byte public preparation and 40-byte borrowed view. Full execution caps remain 128 MiB/33,554,432 work; proof reserve remains 64 MiB/16,777,216 work. Declaration field query remains 20, frozen field query 24 and origin query 72 logical units. Constructor recipe coverage retains 256 scans; both runtime coverage passes retain `256+128I+32F` and supported count 100. Private and ordinary byte-boundary scan schedules remain distinct and unchanged. Any new work is confined to SS physical value traversal or explicitly reviewed existing coarse allowances; no refunds, hidden sidecar or padding use is permitted.

I stage proof, recipes, image correspondence, both preparations, both contexts/executors, import/observation and native target/shape updates together before executable testing. Public verifier/admission already uses the shared map proof, so enabling only its pair guard would be an unsafe partial enrollment. I retain the first four independently annotated originals as the first actual current-consumer gate, then full public/private switch/goto, borrowed repeated import, STRING|VOID/native controls, allocation/output-preservation sweeps and exact old INT/array budgets/output bytes. Publishers and their fresh C-hosted products remain a later separately reviewed phase.

### Invocation-local tag lifetime clarification

I distinguish the temporary preparation local from the value available during execution. For the borrowed public route, an internal preparation helper returns both the owner and uniform tag atomically on success. The existing preparation API wraps it with a NULL tag output, retaining its signature, failure behavior and byte/work totals. The module-taking borrowed supervisor uses the internal helper, keeps its tag on the supervisor C stack and never accepts it from a caller. The preparation and its declarations remain owned and stable for the entire invocation.

The private owner may survive many invocations, so its constructor local cannot supply a later run. Before mutating `busy`, epoch, heap policy, VM error state or roots, `vm_record_map_private_run` independently reads its immutable plan's single pair once through the bounded internal uniform-pair getter. I specify four logical probes per invocation: declaration counts, pair zero, key scalar and value scalar. An explicit local query allowance of four is consumed before those getters; malformed count/pair/tag or failure returns before activation. This is a fixed per-invocation validation schedule, separate from constructor accounting. I do not claim that constructor work prepays infinitely many runs or introduce cumulative guest fuel semantics. No scan over fields/instructions/origins occurs, no allocator is called, and no persistent budget counter is added. Exact old constructor byte/work results remain unchanged; the additional private invocation check is stated rather than hidden.

Both supervisors then pass their local byte as an explicit argument through `vm_rmb_call_root`/`vm_rmx_call_root` to a new internal core implementation, and from that core to `vm_rmb_preflight`/`vm_rmx_preflight` and the map executor. The local survives initializer, entry, CALL/TAIL and every ASSERT-resume core call. It is neither stored in `VmState` nor added to either view. Existing six-, five- and four-argument core wrappers retain their signatures and pass VOID for absent map paths; a wrapper must not authorize a non-NULL map view without the reviewed tag-carrying supervisor. Tests that directly invoke a map core must use the explicit internal seam and test invalid tag/context refusal. Non-map behavior ignores the argument only when both map views are NULL. Every selected preflight compares independent tag, immutable copied row masks, actual operand/header pair and exact opcode effects before mutation. The instructions cannot define their own expected identity.

For observation, output/work must be disjoint from all rooted heap storage as a C precondition. Existing bounded range checks cover owner/path/key spans, not arbitrary heap interiors. The proposed lookup-string call performs full SS validation exactly once before the shared static byte probe; missing/non-STRING result refuses with unchanged output. I check `start<=length` followed by `copy_length<=length-start`, then charge byte work `copy_length+1` and final commit before memcpy. Ordinary lookup still returns a copied VOID observation on absence.
