# Original heap growth accounting prerequisite

Existing retained providers only; no heap/runtime source changes. Final minimal sanitized harness build and execution exited 0 (exact commands and durations in probe.json). The first format and link-global harness errors are retained.

64 distinct STRING keys grow an INT map from 16 to 128 buckets. Reported allocated=2646, freed=7126 after release/collection, objects=0. Difference 4480 exactly equals bucket growth. hm_resize omits growth from allocated stats; release_hashmap counts current capacity. This demonstrates a stats prerequisite, not independent allocation-leak freedom. Future selected-map runtime qualification requires an independent allocator live-payload oracle; no stats correction is authorized here.
