# Union declaration aggregate gate

I merged complete design/baseline, stale map-boundary test correction b91f219fe and declaration implementation2025a38dd as c39c7c4f0. I added test-union-field-declarations to the explicit retirement compiler-contract command; test-units already includes it.

The inspected three-line Make plan rebuilds only ownership_contracts.o and executes the permanent union/map declaration tests against current aggregate dependencies. Union linked/instrumented checks pass (57/491), one method1.625s; measured retained/peak payload112B, four allocations, logical work1152. The corrected map declaration boundary passes131830/135285 linked/instrumented checks, one method1.777s, including12 isolated declaration allocation positions. All68 workflow/checkpoint/capture/container/schema controls pass2.120s. Exact source/object hashes and complete logs are retained.

The component separately retains actual ASan/UBSan qualification, six strict compiler checks, old projection/array neighbors and six byte-identical legacy resource-budget rows. The stale map assertion reproduced against unchanged f535 before correction; linked public verification/native/conversion now agrees with admitted maps, while malformed metadata still refuses. The declaration allocator oracle remains strict and isolated.

This adds a descriptive copied union declaration API only. No union execution, native selection, public transport or paired publisher is enrolled. Fresh platform qualification and the remaining historical proof/recipe/runtime/publisher phases are open. No frontend/bootstrap was run locally for this integration.
