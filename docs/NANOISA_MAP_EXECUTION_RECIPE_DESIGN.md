# I propose copied map execution recipes without admission

This contract follows private proof `854037151`. I propose only an immutable execution-plan constructor and query qualification next. It does not authorize VM handlers, public selection, native translation, image execution or publishers. Public selected-map verification/serialization/native refusal remains a required test. DELETE and all other map pairs remain excluded.

## I give every allocation exactly one owner

I retain `NvmRecordArrayExecutionPlan` unchanged. A separate opaque `NvmRecordMapExecutionPlan` owns two objects: a carrier allocated with that existing type, and one `NvmRecordMapFrozenContext`. The carrier owns its snapshot, instruction rows and fixed function/descriptor arrays. Its `origins` member remains NULL throughout this new construction path. The frozen context exclusively owns its existing base-origin report, map declaration sidecar and map-origin rows. There is no second copy or second owner of the base-origin report.

The old carrier constructor and old queries remain unchanged. The map constructor must not call the old constructor (which correctly refuses map instructions), and must not store a borrowed frozen base pointer into `carrier->origins`. It may reuse static snapshot/allocation helpers, with explicit map-aware preparation helpers. Any helper that needs origin or declaration facts receives an explicit const base-report argument borrowed from the frozen context for that call only. All map wrapper queries route base-origin/declaration requests to that context; carrier queries are reused only where they do not dereference `origins`. No raw carrier or base pointer is exported.

Destruction calls the old carrier destructor, which accepts NULL origins, then the frozen-context destructor, then frees the wrapper. Partial construction uses the same destructor with zeroed members. On every failure the caller's prior output pointer remains unchanged. No mutable analysis, module pointer, budget pointer or callback escapes. Source-module destruction or byte mutation after success cannot alter any query result.

## I specify the next private interface

Proposed declarations in a new private execution header are:

```c
typedef struct NvmRecordMapExecutionPlan NvmRecordMapExecutionPlan;
NvmArrayEligibilityResult nvm_prepare_record_map_execution(
    const NvmModule *, NvmRecordMapExecutionPlan **);
void nvm_record_map_execution_free(NvmRecordMapExecutionPlan *);
bool nvm_record_map_execution_counts(
    const NvmRecordMapExecutionPlan *, NvmRecordMapExecutionCounts *);
bool nvm_record_map_execution_instruction(
    const NvmRecordMapExecutionPlan *, uint32_t, NvmRecordMapExecutionInstruction *);
```

`NvmRecordMapExecutionCounts` contains the unchanged carrier counts plus copied map counts. `NvmRecordMapExecutionInstruction` contains the existing instruction row plus a map operation discriminator and exact key/value/result tag masks. These are new map-only types, not fields added to old objects. Non-map rows have zero map discriminator/tags and an exact unchanged existing recipe. Map rows use a new discriminator (NEW, SET, GET, HAS, LEN), with the existing recipe member set to a documented invalid sentinel, never falsely classifying a map operation as a scalar/array operation. Consumers must select the discriminator before accessing the recipe.

The new API also needs individually typed copying queries matching the existing carrier function, parameter, snapshot bytes/size, descriptor, header, section, metadata and debug queries. Base-origin/count/field/declaration/array-type/binding/variant queries forward through the frozen context's owned base report. Exact map field/origin queries use the existing frozen getters. Every getter preserves output on failure; multi-output aliases refuse. Getters have bounded immutable query costs, with preparation callers explicitly precharging those costs. Existing field24/origin72 bounds remain in force. The wrapper does not expose mutable buffers or a public trusted token.

## I charge simultaneous proof and copies

The map constructor uses the existing execution preparation ceilings of128MiB bytes and33554432 logical work, without raising them. As the old constructor does, it reserves the complete64MiB/16777216 proof allowance before snapshot creation. That allowance includes map declaration live overlap, mutable historical summaries, the frozen context, its owned base report and copied declaration rows: the current map proof already charges all of those inside its own allowance. It is not a free sidecar outside the reservation.

It separately charges the actual wrapper, unchanged carrier, snapshot code/strings/signatures/layout/ownership bytes, instruction rows and decoder temporaries before allocation. The whole proof allowance remains conservatively reserved through final publication even after analysis temporaries are freed. Therefore snapshot plus peak analysis plus frozen report overlap is covered; no credit is reclaimed to fund later copies. The map instruction augmentation is charged by exact row count times actual sizeof, with checked arithmetic. Maximum-stacks output is written into the carrier's existing fixed array by a new private analysis call variant; it is not recovered by a second independent proof pass.

The source-stability comparison covers the original code and every copied logical row, including complete ownership bytes containing kind5. Construction snapshots first, proves the snapshot, builds recipes, compares against the original, then publishes. Old constructor allocations, sizeof values, work charges and six budget receipts must remain byte-identical. New exact budget boundaries, allocation failures, cleanup and independent retries must be measured rather than inferred from these ceilings.

## I cover every instruction and preserve missing GET

The permitted instruction set is exactly the existing record/array proof whitelist plus HM_NEW/SET/GET/HAS/LEN, subject to the current map proof's further constraints. The constructor charges a full256-opcode completeness comparison between supported opcodes and recipe coverage. It decodes all function bodies including unreachable instructions, copies exact operands, verifies boundaries/successors, and retains CALL/TAIL/RETURN continuations and scalar type checks. No map sidecar permits skipping an instruction.

| Map operation | Stack effect | Copied result | Obligations for a later reviewed executor |
| --- | --- | --- | --- |
| NEW | 0→1 | HASHMAP, exact STRING→INT | Exact encoded pair; allocation failure cleanup; publish one new owner only on success |
| SET | 3→1 | Same HASHMAP receiver | Stage map/key/value in original order; exact tags/storage; checked lookup and growth; insertion key retain headroom; preserve receiver alias identity; cleanup popped values on failure |
| GET | 2→1 | INT or VOID | Exact map/key; preserve missing-key VOID; release popped receiver/key; no static presence inference from HAS or literal spelling |
| HAS | 2→1 | BOOL | Exact map/key; checked lookup; release popped inputs |
| LEN | 1→1 | INT | Exact receiver/storage/count; release popped receiver |

Common obligations include first-error cleanup, tags and staged operands where appropriate. These describe checks, not facts discharged by the static proof. GET's copied result mask is INT|VOID even immediately after HAS. Existing tagged-value comparison/type-check behavior remains; no fabricated INT result or zero-on-miss conversion is allowed.

## I retain actual shared-handler semantics as prerequisites

`heap.c:vm_hashmap_set` first looks for an existing equal key. Replacement retains the new value before releasing the previous value and does not retain the new key. For the admitted INT values this performs no heap-value retain. Insertion may allocate a replacement bucket table, rehash, commit it, then search again and retain the stored STRING key. Failed allocation leaves the old table intact; no promise that every possible failure after successful rehash rolls back physical capacity is made. Existing entries are transferred during rehash without repeated key retain/release. Public HM_SET subsequently releases popped key/value and returns the same receiver.

`vm_hashmap_get` returns tagged VOID on absence. Public HM_GET retains the result before releasing receiver/key; exact INT|VOID needs no heap-result retain. These distinctions require future duplicate-interned-key, replacement, growth-failure, headroom and first-error runtime tests. This recipe phase does not introduce a work counter into lookup/growth or change shared handlers. No runtime-budget policy is authorized by copying a recipe.

## I qualify this phase separately

Required tests are complete old/new opcode recipe coverage; exact stack and GET result masks; copied source/function/layout/constant identity after original destruction; field/origin query lifetime and alias failures; all allocation prefixes and sentinel preservation; byte/work exhaustion plus recovery; four retained actual anchor modules privately annotated only inside the test; original and public refusal unchanged; and six old budget receipts unchanged. No actual executable canonical roundtrip is claimed while public readers intentionally refuse map authority.

A subsequent copied image/runtime contract must choose independent snapshot/context copying with a single owner for each base report, validate all referents again, and charge execution-plan plus image overlap. It must specify physical map checks, retain headroom, lookup/rehash accounting, private/public invocation lifetime, native key closure and HM_GET tagged semantics before any admission change. This document does not implement or authorize that phase.

## I retain lookup topology as a later physical invariant

`hm_resize` transfers occupied entries and assumes unique keys and valid lookup topology; it does not reject an equal key found while rehashing. A later physical validator must prove occupied keys are unique and reachable under the actual probe sequence, or establish those properties from closed construction/mutation. Tags/count/capacity alone are insufficient. Duplicate equal keys or unreachable occupied entries cannot enter a selected runtime through an observation/import shortcut. This prerequisite adds no runtime traversal or policy in the present recipe-only slice.
