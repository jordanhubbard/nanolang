# I qualify descriptive mixed layout identities

I qualified source/test pin `1e2df4851855a5bcc50d5703f5e8b83034d986e3`, including
reviewed production67e57e32. My [sealed reports](mixed-layout-view/manifest.json)
retain commands, outcomes, before/after inventories and compiler selection.

* GCC focused gate passed in 4.877 seconds: 138 descriptive checks.
* Clang focused gate passed in 0.715 seconds: the same 138 checks with
  AddressSanitizer, UndefinedBehaviorSanitizer and leak detection.
* Retained layouts, ownership contracts, verifier and managed-record-plan gates
  passed together in 16.609 seconds, including all 96 verifier tests.
* All 1,602 inventoried source/test/build files remained unchanged across gates.

My focused reader, layout decoder, cursor and C fixture were rebuilt with sanitizer
flags; other linked library objects retained the ordinary build. I do not claim
fully instrumented VM coverage. Allocation injection covers calloc/malloc in those
three rebuilt source units, including descriptor-copy allocation; the sweep reaches
success and preserves the previous output and input bytes at every failed budget.
Temporary focused executables are removed by their harness; I retain their source,
flags and measured outcomes, not a claim of archived executable hashes.

I cover exact source/global/compact maps, pending nested arrays, both wire versions,
path bytes, owned storage after source disposal, nested unborrowed resource positive,
both nested borrowed-mode refusals, incomplete missing-edge UNKNOWN versus COMPLETE
invalid, indexed incomplete-child propagation, malformed unused fields/padding,
limits and prior ordinary/shared-validator decisions. No pending mixed module was
executed. Existing adjacent tests execute only their own previously accepted controls.

My static prereview correction is preserved in c8cbe46f/67e57e32; no pre-correction
query or module was executed. These results do not admit mixed execution, Samples,
managed owner fields, source syntax, or the full product. Task20d4 stays open until
canonical merge; parent4be and full ownership acceptance remain open.
