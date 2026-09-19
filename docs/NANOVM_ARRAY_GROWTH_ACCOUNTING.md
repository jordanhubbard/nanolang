# My VM array growth accounting

I track this prerequisite as `task_50c8c460a5994dd3b8a20180bd287391`, required by private owned ARRAY runtime parent430220. I do not change public admission or array semantics.

## Retained finding

I preserve frozen78b9's [first terminal](evidence/private-owned-array-runtime-graph-first.json). GCC case0, fused0 returns success. My exact graph audit finds only module STRING roots and one buffered zero-reference STRUCT, then collection restores the original object count. The byte-baseline assertion fails afterward. I did not log its post-collection numeric byte value, so I do not claim a measured discrepancy or a product leak.

Static `src/nanovm/heap.c` inspection shows `array_grow` changes capacity after successful reallocation without updating `stats.allocated`; `release_array` adds final-capacity storage to `stats.freed`. My ordinary FLOAT fixture grows capacity8 to16. The missing growth charge is statically64 bytes for that growth, not a measured terminal difference.

## Bounded correction

I pass the existing `VmHeap*` from `vm_array_push` into private `array_grow`. After successful reallocation, before publishing the new capacity, I charge `(new_capacity-old_capacity)*element_size` to allocated bytes. Existing size/overflow guards precede reallocation; failed growth changes neither counters, capacity, storage pointer nor contents. I retain existing allocation_calls semantics, reference handling, capacity policy and public signatures. I do not claim complete accounting for other heap families.

## Acceptance

I review production before fresh fixtures. Ordinary unboxed FLOAT and boxed STRING arrays must charge exact successful growth deltas and return object/byte baselines after release/collection. A controlled required-capacity realloc failure must leave storage, capacity, contents and accounting unchanged and permit later successful recovery. I keep the private runtime graph audit, extra-root negatives and original byte assertion unchanged; diagnostics will print post-GC allocated/freed/live totals before that assertion. I preserve every earlier terminal and run fresh artifacts only. Strict GCC/Clang sanitizer controls and existing heap/provider adjacency precede resuming the full private runtime gate.
