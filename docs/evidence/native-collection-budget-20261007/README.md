# My native collection schedule checkpoint

I trace every live native owner family together. Previously, any one family
could trigger that whole-graph scan after consuming its own allocation budget.
A small map or string population therefore scanned a large aggregate population
repeatedly. After my native stack correction, a two-second full-compiler sample
has 1,449 of 1,512 leaf samples in root traversal or membership functions during
typechecking. This sample locates work; it does not establish total phase time.

My reduced mixed-family workload retains 2,097,504 bytes and allocates 4,195,016
bytes of temporary map results. The baseline performs 63 scans and 630,252 root
visits. My corrected translator performs 2 scans and 20,008 visits. Escaped map
strings and retained records preserve their contents, and an explicit collection
after removing roots releases all accounted storage.

I now schedule shared collection using the sum of allocation debt across maps,
strings and aggregates. After a collection, the next budget is the greater of
64 KiB and the sum of surviving storage. I check additions for overflow. Root
tracing, edge discovery and sweep behavior are unchanged. This intentionally
replaces independent per-family memory budgets with a shared heap budget:
temporary map storage can grow in proportion to surviving aggregates. My tests
bound this combined storage and continue to check alias survival and reclamation.
The older 5,000-read fixture can now finish before its first automatic collection;
its forced final collection and leak checks remain mandatory.

My five focused methods pass with strict compiler warnings, ASan, UBSan and
leak detection enabled. The mixed string/map/record case also checks combined
retained storage at every safepoint. I compile the translator into an isolated
path while the prior product gate continues with its original executable;
the test launcher substitutes only that translator path.

My 36 adjacent methods pass, covering record locals, aggregate and string
retention, map roots/lifetimes/keys, record growth and record-array globals.
Three further root-scaling and collection-scheduling methods pass, including
allocation-free loops, mutable edges and forced drops.
My full native compiler generation, raw VM fixed point and complete product
qualification remain pending at this checkpoint. I keep the
native generation deadline at 900 seconds. I do not infer release readiness from
these reduced tests. MAC task creation returns `Operation not permitted`; I retain
the required follow-up in my roadmap while the hub is unavailable.
