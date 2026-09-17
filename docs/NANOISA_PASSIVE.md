# My scalar passive eligibility record

I carry version-1 scalar eligibility records in v2 section `0x0c`, guarded by
feature bit `0x40`. Readers that do not know that feature reject the module.
Legacy output refuses records rather than dropping them. This is a bounded IR
foundation for `PASSIVE_PARALLELISM_DESIGN.md`; frontend syntax, call summaries,
resource metadata and the full conformance matrix remain unfinished.

Every field is a little-endian `u32`. The section starts with `version = 1`
and a positive block count. Blocks follow function index and entry offset order,
without overlapping ranges. Offsets are absolute positions in the code section;
exit offsets are exclusive. Each block contains:

- kind (`1 = par`, `2 = flow`), function index, entry, exit, node count;
- one node per source binding, in source order; its array index is its dense ID;
- each node: entry, exit, result local, dependency count, external read count,
  observable effect mask (zero), resource count (zero), sorted dependency IDs,
  then sorted external parameter local indices.

No trailing bytes, duplicate IDs, unknown version, flags, effects or resource
claims are accepted. A node ID also defines its source-order index; there is no
second field that can disagree with it. I recompute the lowest-source-index
ready node at every step. Physical instruction ranges follow that stable
**topological** order, including forward references in `flow`. `par` admits no
internal edges. Ranges exactly partition the block, with no gaps or overlaps.
The node's final instruction stores its result, which serves as the serial
binding commit; the next node begins with an empty stack.

This first verifier permits straight-line scalar stack operations and arithmetic.
Each node may read declared completed node results and may write only its
distinct result local. External read counts must be zero in this version: the
ordinary verifier does not yet prove caller values from parameter annotations. It cannot access
an incoming stack value. Every recorded read/dependency must occur in code,
and every code read must be recorded. Result locals must not alias parameters.
External input support is tracked by `task_bf571298c10d4cc5a387b9f233ff3c40`.
Calls, captured values, aggregates, host operations and
handler control flow are refused in this slice. Outside node ranges I also
allow ordinary branches, return/halt, print and assert; a branch cannot enter
the middle of a block. These restrictions are explicit prerequisites, not a
claim that scalar-only functions exhaust the language contract.

Ordinary verification still applies. Serial VM and native translators execute
the unchanged instruction stream. Loading, verifying, or serializing malformed
eligibility claims fails; metadata-free modules retain their existing behavior.
The bridge copies the exact payload. Code transforms that change offsets must
rebuild valid records; they cannot retain stale claims or silently discard them.
