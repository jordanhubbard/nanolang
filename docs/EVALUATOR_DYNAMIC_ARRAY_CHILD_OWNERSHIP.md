# Dynamic array reference leaves

I audit exact source `dfbf20342f0577fcab316b21c04bb3bee9a35e11` before changing evaluator-created dynamic-array children. This work remains under task_992713bde1494772b0cb0b58bc9ee3c3. No affected dynamic-child test has been run to reproduce the missing owner.

## Existing contracts and allocations

My low-level `DynArray` stores borrowed child pointers. `GC_TYPE_ARRAY` destroys its data buffer and header, not pointed-to strings or records. I preserve this contract, including caller-created inputs and borrowed reverse/conversion/nested-array aliases. I do not adopt a pointer merely because it appears in an array slot.

The evaluator separately creates fresh children without a destruction owner in these routes:

| Route | Fresh allocation |
| --- | --- |
| Three dynamic arithmetic helpers | `malloc` concatenated string per result element |
| Dynamic `array_set` | duplicated string or cloned record |
| Dynamic `array_push` | duplicated string or cloned record |
| Dynamic `array_slice` | duplicated string or cloned record |
| Dynamic `map` and `filter` | duplicated string |

`str_split` already allocates its character, segment and tail strings with `gc_alloc_string`. I leave those GC allocations unchanged. The standalone `at`/`array_pop` result-copy repair is separately reviewed at dfbf and does not own stored dynamic children.

## Actual consumers

I inspect the exact interpreter `call_function` consumers, excluding unrelated VM APIs with suffixed names. `nano_main` and `wasm_interface` consume the scalar main result before freeing the Environment. DAP and bench discard their result with the Environment still alive. Property tests inspect their result synchronously; evaluator File callbacks also operate while their Environment is live.

The dynamic map-result fixture reads string elements before `run_ctx_free`; integer and binary64 fixtures use scalar dynamic buffers and preserve their explicit `gc_release` contract. The loop-string fixture explicitly frees fresh dynamic replacement strings at test_eval.c2597: this is a real consumer that must change with the ownership policy. It separately owns the original input strings, whose frees must remain. Its escaped final result is an independently copied public string, not a dynamic-array leaf. The projection fixture similarly owns its original caller-created dynamic record pointers and must retain their disposal.

I find no inspected in-tree consumer that reads an evaluator-created dynamic string/record leaf after Environment destruction. This is a bounded source audit, not permission to invalidate caller-owned input leaves or independent public record/string snapshots.

## Proposed correction

I give each fresh evaluator-created string/record leaf an independent snapshot in the existing Environment result owner before publishing it into a dynamic slot. The snapshot helper retains its existing transactional failure rule. Numeric GC array headers/buffers and the low-level runtime ABI remain unchanged. On overwrite I do not free the old pointer: it may be a caller-owned input or an alias; independently owned evaluator snapshots remain in their cumulative Environment arena until teardown.

I pass the actual Environment through the three recursive dynamic arithmetic helpers and the mutation helpers that currently lack it. Temporary concatenation buffers are released after an independent owned snapshot succeeds, and on snapshot failure. No borrowed reverse/conversion/nested pointer is registered. Existing GC-managed split strings are not registered a second time.

I state explicitly in the public header that evaluator-created reference leaves inside a dynamic array borrow the Environment even when the GC array buffer has a separate reference lifetime. A caller-created leaf retains its caller owner. Existing task contextual leases retain the Environment for completed reference-bearing results; no new task transport or ownership inference is introduced.

## Required controls before acceptance

I preserve original collection/projection assertions and add real string/record dynamic set, push, slice, string map/filter, and all three concatenation routes. I check exact text, independent copies, mutation aliases, old caller-leaf preservation, GC-buffer release before Environment teardown, and a completed task result with dynamic reference leaves retaining its owner. I remove only the loop fixture's frees of evaluator-created replacement strings; original caller input frees and numeric GC releases remain.

Source and fixture review precede fresh ordinary and ASan/UBSan/LSan owning qualification on both hosts. I do not infer complete evaluator or CI acceptance from those focused results.
