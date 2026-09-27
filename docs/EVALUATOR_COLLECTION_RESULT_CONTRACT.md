# My evaluator collection result contract

I plan this boundary under task_992713bde1494772b0cb0b58bc9ee3c3 before registering collection owners. My contextual completed-task lease/cross-Environment prerequisite must pass its owning gates first. This document and the linked cleanup proposal do not activate destruction.

## Public return identity

My public call_function transfers no ownership of returned fixed VAL_ARRAY or interpreter HashMap identities. A fresh evaluator allocation borrows its Environment. An alias of a caller-created input retains its original caller owner; I do not adopt it. Each actual owner must outlive its borrowed references. Returning an alias cannot detach or transfer a live collection.

My public independently cloned record/tuple snapshots, string/callable results and VAL_DYN_ARRAY GC references retain their existing contracts. Nested collection leaves in copied aggregates are still borrowed. Public C callers must keep the actual Environment or caller owner alive while using them; this is not a new automatic snapshot lease. Completed task slots preserve their source Environment under the separately reviewed contextual lifetime contract. My caller audit found no direct public consumer expecting nested Array/map use after Environment teardown; the existing escaped lifecycle fixture contains independently copied record/string scalars only. HashMap currently has VAL_INT representation, so a declared HashMap return must not enter a scalar task-result fast path solely because its runtime tag is INT.

## Caller inventory and cleanup proposal

I inspected all17 repository C/header files containing130 textual call_function references at53cc; [my exact inventory](evidence/evaluator-collection-owner-plan/call-function-consumers.json) records source hashes and lines, including declaration/comment/definition references.

| Consumers | Collection consequence |
| --- | --- |
| nano_main, wasm_interface, dap_server | Main results are scalar or ignored; Environment destruction remains after use. |
| bench_native, proptest | Results are ignored or inspected before Environment destruction; existing string cleanup is separate. |
| eval_io result_map/result_and_then | Callback collection leaves can escape inside unions; they depend on contextual owner lifetime rather than local array destruction. |
| eval.c internal callbacks, task publication, REPL wrapper | Internal aliases and nested returned leaves retain owner context; root/foreign-task checks must precede collection teardown. |
| test_eval mapped-return controls | Remove only manual destruction of evaluator-created fixed-array output/owned string slots; keep all value/type/input assertions and caller-created input cleanup. |
| integer negation/binary fixtures | Separate fixed returned-result cleanup from independently created input cleanup; retain existing dynamic GC release and every arithmetic assertion. |
| aggregate binary64 fixture | Returned fixed arrays are observed without manual release. Its independently created fixed inputs still need their own cleanup audit; registry adoption is forbidden. |
| binary64 arithmetic/bits, benchmark, File-parser fixtures | Scalar or dynamic results preserve existing contracts. |
| evaluator lifetime programs | Existing escaped record/string-only snapshot check remains independently owned; no collection adoption is inferred. |

My [proposed header and three-fixture diff](evidence/evaluator-collection-owner-plan/cleanup-proposal.patch) is not applied. It must land only with the reviewed registry implementation and matching source/fixture qualification; executing it alone would remove the old manual cleanup before a new owner exists. It preserves all assertions, caller inputs and dynamic-result releases.

## Required implementation order

I first qualify contextual result ownership, including declared non-scalar map handles with runtime INT tags and borrowed external collection inputs. Then I normalize map_keys/map_values string slots to independent copies with transactional cleanup. I register only complete evaluator-created fixed arrays/maps and define explicit detach-before-map_free and partial-construction rollback. Each registry entry destroys exactly its owned allocation, not borrowed nested arrays or caller inputs.

I must qualify fresh/static/nested/shared aliases, public return lifetimes, callback early return, map projection mutation independence, explicit free, allocation refusal/recovery, and actual owning full test-eval. My earlier arithmetic/compiler qualifications remain scoped to their recorded source; I do not reinterpret them as registry acceptance.

## Qualified prerequisite and exact publication policy

I base this isolated implementation on `73e86c9637e25fea652aea0dee3ded5386f981ff`, whose contextual task lifetime gates passed ordinary and sanitizer configurations on both hosts. Evidence-only `8c6b37bda` seals those results. The separately proposed union registry is not included or claimed qualified.

I refine the earlier complete-value registration plan: I may register a fixed array immediately after its header, element type, length, capacity and zeroed storage are initialized. Every destructor-readable pointer slot is then either NULL or an independently owned value written later. Scalar/nested-reference slots need no destructor. This gives partial refusal paths an eventual Environment owner. Existing map/literal callback-return rollback must explicitly detach before its current partial destructor runs. No entry may survive destruction of its allocation.

My private owning constructor checks storage/header/registry allocation before publication. A fatal registry allocation failure first destroys the new unpublished allocation, then reports the exact failure and exits; it never destroys borrowed caller inputs. Low-level public `create_array` remains caller-owned and unchanged. Maps register only initialized map allocations; explicit `map_free` detaches before the existing map destructor. Projection arrays duplicate every string and roll back only their own completed copies on failure.

I will test every allocation prefix with exact fatal refusal and fresh recovery processes; NULL-safe partly filled string/record slots; borrowed caller arrays and aliases; callback early-return graphs; map projection mutation independence and explicit map release. Full original test-eval remains a final integrated-source obligation. I do not run teardown changes before independent source/fixture review.

My fixed-record `array_set` currently clones and overwrites the old owned slot without destroying it; `static_array_remove` already destroys that same slot ownership. I must clone the replacement successfully first, then discard the prior record, including the case where source and destination alias. Registry destruction alone cannot reclaim an overwritten pointer. This is a static finding, not a reproduced failure.

Registration is O(1) and is called exactly once at each actual fresh-allocation site. Bindings, aliases and public returns never call it. The entry carries its exact destructor kind. Detach searches by pointer only at explicit destruction/rollback boundaries; I do not impose a growing linear duplicate scan on every parser/evaluator allocation.

My owning fixed-array helper accepts the original signed64-bit requested length/capacity and rejects negative or greater-than-INT_MAX extents before assigning the Array's int fields. Map projection size reaches this check without a prior int cast. My string array_set duplicates/checks the new string before freeing its old slot, preserving actual aliased input leaves and unchanged storage on copy failure. These are static review findings; I add boundary/alias tests without executing the old paths.

## Additive owning fixture

My focused fixture retains the existing partial-literal, higher-order handler, empty-alias and nested-array assertions. New direct evaluator controls exercise external input slicing without adoption, owned string/record alias replacement, independent slices, map projection copies after map destruction, public call_function aliased-string input and oversize array/map projection refusal before narrowing. Parsed string/record literals allocate again while their first owned slot is populated; each measured new-allocation prefix runs in a fresh transient/persistent-failure process, then a fresh successful recovery. The allocation hook covers only new owned-array header/storage, owner entries and copied projection/replacement strings; existing parser/record-cloner allocators are not interposed. Atexit cleanup verifies caller input bytes before and after Environment destruction, with exact fatal diagnostic/exit status required. I retain whole-process sanitizer checking, without suppression. The retained source callback controls require integration with the independently repaired shared evaluator/checker prerequisites before execution; this isolated base is not a whole evaluator acceptance claim.

My additive task controls use the existing real evaluator bundle/await adapters from `test_evaluator_lifetime_eval.c`, now compiled into the collection test object. Actual parsed functions allocate a fixed integer array and an int-encoded HashMap. After each task reaches DONE and drops its argument bundle, I release an explicit caller Environment lease, assert the result lease still blocks destruction, read the registered result, release the task and only then destroy the Environment. This directly exercises the new registered allocations through the qualified contextual lifetime boundary.
