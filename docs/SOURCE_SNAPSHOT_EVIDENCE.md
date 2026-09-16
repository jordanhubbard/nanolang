# Native Source Snapshot Evidence

I resolve each native source once when I record and validate its content hash. I preserve absolute paths and join only relative paths to the module directory. A missing source, a failed read, a zero digest, or a path that does not fit my bounded buffer cannot become cache evidence.

`tests/test_module_builder_cache.c` checks relative and absolute digest equivalence, reuse of an unchanged absolute source, invalidation after changed bytes, failed reads, and overflowing paths.
