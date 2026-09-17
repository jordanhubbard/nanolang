# My native root lookup cost

I recorded MAC `task_869e7e8e12e946d2a3ffc9cac6e16882` after a native compiler self-compilation sample spent CPU in `nroot_add`, `nroot_trace`, and `nmap_collect` while splitting merged source. My old collector linearly searched all prior roots on insertion and scanned all owners for each string/map root. This made a collection quadratic as the live graph grew. The slow sample did not demonstrate a correctness failure.

I now keep my traversal list in insertion order and index `(storage kind, pointer)` membership with a half-full open-addressed table. Table entries are indices, so growing the traversal list does not leave stale pointers. I clear both count and membership when publishing a fresh frame snapshot, and release both allocations when a frame returns or collection completes. After traversing the same graph, I scan each owner once and check membership before the existing sweep.

I retain every collection point and every aggregate edge. I do not cache flattened mutable-array contents, suppress collection, change root tags, or change the sweep policy. Cycles terminate through identity deduplication, and distinct storage kinds at the same address remain distinct roots.

My sanitizer regression builds an aliased array containing two references to each owned string. It counts real identity comparisons rather than asserting host-dependent timing. It also checks root reset/reuse, cyclic records, live string contents, and reclamation after dropping the array edges and final map root.

| Distinct strings | Previous comparisons | Indexed comparisons, observed run |
| --- | ---: | ---: |
| 1024 | 2101250 | 4514 |
| 2048 | 8396802 | 8921 |
| 4096 | 33570818 | 16726 |

Pointer hashes vary with allocation addresses; my acceptance bound is 64 comparisons per distinct string for this workload. The original implementation fails that bound. I rerun the same test against an unchanged translator through `NANO_ROOT_SCALING_TRANSLATOR`, so the before/after workload remains reproducible. At 4096 strings, the observed sanitizer trace CPU time decreased from 0.040681 seconds to 0.000604 seconds. These timings describe this workload and host, not a universal speedup claim.

My native translator passed 2386 checks, shape constraints passed 1092 checks, and the returned-array/string cleanup regression passed. My focused scaling, map lifetime/global, and string-join suite passed 15 methods, with sanitizer checks covering surviving aliases and bounded owner counts. The scaling regression enables AddressSanitizer, UndefinedBehaviorSanitizer, and leak detection. Existing ownership gates retain their documented leak-detection settings.

I preserve the original full compiler executable and its progress log. The revised translator produces a separate executable from the same pinned compiler bytecode for comparison. That full self-compilation and byte-identical convergence remain separate acceptance criteria; this change alone does not prove them.

Local evidence: `/tmp/nanolang-native-root-cost-baseline.log`, `-focused.log`, `-scaling-final.log`, and `-native.log`. My regression is part of `make test-one-ir-compiler`.
