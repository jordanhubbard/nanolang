# My GC pointer-entry experiment and allocation-size defect

I measure a private candidate that reserves aligned pointer-table entry storage
after the managed payload, preserving `GCHeader` layout and logical statistics.
Each of three executions allocates, looks up, retains and releases 100,000
objects in batches of 1,000 across fifteen sizes from zero through 4,097 bytes.
It checks pointer identity, header sizes, refcounts, finalizers, foreign pointers,
zero final managed counts, shutdown and reinitialization. I retain actual source,
commands, unchanged input/product hashes and exact output.

| Mode | Separate entry, two allocations (CPU seconds) | Private tail entry, one allocation |
| --- | --- | --- |
| Plain O2 | 0.008965, 0.005958, 0.005151 | 0.007671, 0.005017, 0.004316 |
| ASan/UBSan | 0.031592, 0.031133, 0.028474 | 0.031963, 0.022757, 0.020874 |

I do not apply the candidate. My additional actual hash-collision control
performs a live finalizer pointer lookup during shutdown. The production
separate-entry implementation passes; the private tail candidate leaves a freed
entry reachable and fails with an ASan heap-use-after-free. I retain both exact
sources, commands and diagnostics in `shutdown-collision-*.json.gz`. These timings
do not qualify a whole compiler or excuse a lifetime failure.

I separately reproduce #1118 in the current production allocator:
`gc_alloc(SIZE_MAX, GC_TYPE_OPAQUE)` wraps the header-plus-payload size, allocates
a short block and writes past it. `overflow-baseline.json.gz` retains the actual
ASan heap-buffer-overflow, source/product hashes and commands. I correct only
the overflowing size arithmetic in production, refusing it before allocation
or registration. I retain the separate pointer-table allocation and unchanged
header layout, logical accounting and valid allocation paths.

My existing GC-struct unit gains a boundary control alongside its eleven original
controls: extreme refusals preserve statistics, fifteen valid sizes retain exact
logical size and ownership, and foreign pointers remain unmanaged. All twelve
pass under local ASan/UBSan. Full current platform/runtime qualification remains
required; the private performance candidate and frame-carrier reclamation are
separate.
