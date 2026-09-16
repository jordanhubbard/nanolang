# My nested-record resource classification

I replace the direct `is_resource` flag lookup in `is_resource_type` with a
least-fixed-point computation across registered records. A record inherits
the obligation from a resource-bearing record field. Cycles without resources
remain ordinary; cycles reaching a resource become resource-bearing. There is
no recursive call depth cutoff. Field names resolve in the containing record's
module, and I restore the caller's module context afterward.

The temporary classification array is freed after each query. If allocation
fails, I report the failure and conservatively return resource-bearing rather
than silently omit an obligation. Allocation-failure injection is not covered
by this test. This helper uses the existing sequential environment lookup;
it does not add concurrent typechecking support.

`make test-resource-classification` exits zero. It checks a 300-record cycle
before and after marking its last member as a resource, direct resources,
missing/null names, and conflicting field-type names in two modules. The
caller module remains unchanged. The gate is included in `test-units`.
Host-local log: `/tmp/nanolang-resource-classification.log`.

This is a dependency for ownership analysis, not completed analysis. Union,
tuple and collection propagation, path-sensitive moves, branch joins,
borrows and frontend integration still require work. The C-seed parity
failures from `013f7905` are not claimed fixed. MAC
`task_91ae827be4154eaa8f22698aeecc8cf1` remains the implementation task.
