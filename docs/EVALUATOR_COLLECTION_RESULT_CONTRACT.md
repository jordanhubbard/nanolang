# My evaluator collection result contract

I plan this boundary under task_992713bde1494772b0cb0b58bc9ee3c3 before registering collection owners. My contextual completed-task lease/cross-Environment prerequisite must pass its owning gates first. This document and the linked cleanup proposal do not activate destruction.

## Public return identity

My public call_function transfers no ownership of returned fixed VAL_ARRAY or interpreter HashMap identities. A fresh evaluator allocation borrows its Environment. An alias of a caller-created input retains its original caller owner; I do not adopt it. Each actual owner must outlive its borrowed references. Returning an alias cannot detach or transfer a live collection.

My public independently cloned record/tuple snapshots, string/callable results and VAL_DYN_ARRAY GC references retain their existing contracts. Nested collection leaves in copied aggregates are still borrowed: completed tasks and public snapshots must preserve the source owner under the separately reviewed contextual lifetime contract. HashMap currently has VAL_INT representation, so a declared HashMap return must not enter a scalar task-result fast path solely because its runtime tag is INT.

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
