# My allocation-driven native collection

I recorded MAC `task_5928905033844a4a8fb498d63cb68c39` after indexed root membership still left the full native compiler repeatedly scanning source arrays in `split_lines`. A read-only sample of the indexed compiler found `nmap_owned_live = 1` and `nmap_owned_peak = 1` while the collector traversed the growing source graph. That sample demonstrated redundant work, not a correctness failure.

I now record allocation debt whenever I create either kind of collectible owner: a map or a copied string returned from a map lookup. My existing backward-edge and self-tail safepoints still publish current frame roots. They scan only when allocation debt exists. A scan closes the current graph, performs the existing sweep, and clears the debt. An explicit `nmap_collect` remains an unconditional collection.

Without a new owner, the owner count cannot increase. Dropping the last reference during such an interval can defer reclamation, but the retained owners are bounded by the set that survived the preceding collection. The next allocating safepoint traces current mutable array/record contents; it does not reuse a flattened graph from an earlier scan. Program exit still releases all remaining owners. I introduce no collection batch threshold and retain existing bounded-churn assertions.

My real NanoISA fixture allocates a map and copied string, executes 5000 allocation-free loop iterations, changes the string owner, then repeats the loop. I count actual calls to root tracing in the generated C:

| Collector | Full scans | Peak owners | Owners after exit |
| --- | ---: | ---: | ---: |
| Unchanged indexed collector | 9998 | 3 | 0 |
| Allocation-debt guard | 2 | 3 | 0 |

The same regression can use an unchanged translator through `NANO_DEBT_TRANSLATOR`; it fails the scan-count assertion after clean teardown. I do not compare wall-clock time in this acceptance assertion.

A second ASan/UBSan/LSan regression removes the last string edge, runs 10000 guarded safepoints without allocation, then replaces the edge in the same caller-owned array with a newly allocated string. It requires the new collection to keep the replacement, reclaim the old owner and unrooted map, and preserve the previous owner ceiling. It separately forces collection without debt and verifies that forcing also clears pending debt.

My map/global/string/scaling regressions retain their existing lifetime, content, peak-owner and sanitizer assertions. My full compiler comparison uses a separate executable translated from the exact same pinned compiler bytecode; the original and indexed baseline executables remain unchanged. Full self-compilation and byte-identical bootstrap convergence are separate acceptance criteria.

Local evidence uses `/tmp/nanolang-native-collection-debt-{baseline,focused,guarded,native,returned,products}.log`. The generated baseline/debt executables and stage-comparison logs remain under `/tmp/nanolang-bootstrap-8d769-*` and `/tmp/nanolang-native-collection-debt-seed*`. My durable regressions are in `tests/test_native_collection_debt.py` and run in `make test-one-ir-compiler`.
