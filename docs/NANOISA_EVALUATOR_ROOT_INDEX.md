# My exact evaluator root index

I track task_c2e9d2f19f1b4a359e841edcceda4abd under timing prerequisite
task_2deaad56f65c497f80546220aa1ca9d0 and my required list/evaluator work.
This is a preimplementation design. My production source remains cc60606c9.
My separate diagnostic branch is sealed at 82f339a25; I carry its exact report
blobs here without adding its observer to production.

## My measured reason and limits

My completed diagnostic interval contains 353,202,393 linked-root visits in
28,655 membership calls: 1.634 seconds out of 1.989 measured seconds. The final
parse_block_recursive shadow was killed without an end marker. I do not attribute
that remaining interval or claim the full deadline will pass from this measurement.
The original complete shadow graph, ten-second limit, nine-method matrix and
unchanged full Make acceptance remain required.

## My identity and lifetime

I replace only env_record_result_borrowed's membership search. Membership is the
exact pair (ValueType, root pointer), for VAL_STRING, VAL_STRUCT and VAL_TUPLE.
A nested record/string/tuple pointer is not a root merely because an indexed
owned graph contains it. I do not index list handles, symbols, arbitrary aliases,
callable metadata or values belonging to another Environment.

Each Environment owns one private EnvRecordIndex pointer, initially NULL through
its existing calloc constructor. The private index allocation has a size_t
capacity, size_t count and a flexible array of pointers to EnvRecordResult entries.
Its slots borrow entries from that Environment's existing monotonic arena list.
The list still owns every graph and remains the destruction order. I do not free
roots sooner, change source authority or infer contents from membership.

I audit every Environment constructor/copy and every record_results assignment.
Only env_value_snapshot and env_retire_value publish entries today. No accepted
path removes a published entry before env_record_storage_free. Scheduler leases,
provider generations, public result clones and teardown refusal remain unchanged.
My transaction has no evaluator/user callback; it neither crosses an await nor
stores shared thread/global state. Allocation instrumentation is trusted test code.

## My checked table and lookup

I use open addressing with linear probing and a mixed uintptr_t pointer/type
hash. Hash collisions never decide identity: both the actual pointer and ValueType
must match. Every probe is bounded by capacity. An empty slot ends a miss. The
index requires neither a tombstone nor deletion because its roots are monotonic.
Independent environments never share index slots or ownership.

Capacity starts at 16, stays a power of two and grows before insertion would
exceed capacity minus capacity/4. I check count+1, capacity doubling and
sizeof(EnvRecordIndex)+capacity*sizeof(slot) before allocation. Capacity/count
arithmetic cannot wrap. This adds a checked allocation failure, not a fallback
that silently loses membership. Lookup itself does not allocate or mutate.
I claim expected constant lookup under this distribution, not a collision-free
hash or an adversarial worst-case constant bound.

A duplicate (type,pointer) publication returns false and publishes nothing.
Retirement already refuses an existing borrowed root; I preserve that behavior.
Repeated snapshot requests clone new independent graphs and therefore may publish
new roots. Duplicate rejection does not free an existing indexed graph.

## My publication transaction

I introduce one internal publication helper; I do not add a public lookup API.
Its entry argument is a complete but unpublished EnvRecordResult. Ownership
transfers to the Environment only on true.

1. I retain the existing entry allocation and checked graph clone for snapshots.
   For retirement, the caller still owns the supplied graph at this point.
2. I check supported root type/pointer, duplicate identity, count and capacity.
3. If growth is needed, I allocate a separate zeroed table and rehash pointers to
   every old entry into that table. The old table and arena remain untouched.
   Rehash never clones, frees or modifies an entry. I locate the new entry's
   vacant slot before committing. Without growth I locate an existing vacant
   slot, also without publishing.
4. After every fallible action succeeds, I publish the new table if any, populate
   the prepared slot, update its count, and link the entry at the arena head.
   These operations cannot allocate or call user code. I release the old table
   storage after replacement; its entries and graphs remain owned by the arena.
5. On any failure, I release only new table/entry storage and a newly cloned graph
   when applicable. Snapshot output remains unchanged. Retirement's supplied graph
   remains caller-owned and unchanged. Old membership, count, capacity, roots and
   aliases remain unchanged; there is no partially published new root.

The caller clears a retiring symbol only after true, as before. On Environment
teardown I detach and free the index, then release every arena graph and entry
exactly once through the existing owner. No lookup can retain an index pointer
past teardown. My active-lease refusal still runs before this destruction.

## My fixtures and acceptance

I extend the actual env.c allocation-hook fixture, not a modeled table alone.
Measured persistent-prefix and transient sweeps include first index allocation
and growth after 12 occupied roots. Each failure must preserve the sentinel
output or retirement input, every old lookup and alias, old index/list state,
and allocation balance; a fresh recovery follows. A successful growth reuses the
same old entry/graph pointers. I retain all existing cloning and list sweeps.

I use valid live allocations to produce colliding buckets and require independent
lookup of each exact key; bounded probing must find a missing key without hanging.
I test repeated retirement refusal, equal text at different addresses, different
supported root types, a nested pointer that is not a root, two independent
Environments, empty queries, capacity/product limits and teardown exactly once.
I do not dereference fabricated pointers or execute invalid memory accesses.

I update old hardcoded allocation counts only where the newly required index
allocation changes the actual checked transaction, retaining the old assertions.
I submit production first and complete fixtures second for independent review.
Then I rebuild all providers from the new header/source on both hosts before the
original full build/bootstrap and focused/source/neighbor gates. Prior binaries
cannot stand in for this changed Environment layout. My enum and full Make tasks
remain open until their own acceptance is established.

## My production checkpoint audit

I implement the index only in src/env_record_lists.inc and add its opaque pointer
in Environment. The constructor search finds one allocating production constructor,
create_environment's calloc; no production by-value or memcpy Environment copies
were found. Four existing stack Environment fixtures use zero/designated initializers
and do not publish arena roots. All consumers must rebuild against the new header.

The publisher search finds only env_value_snapshot and env_retire_value. Both now
use record_result_publish; the only other arena-head update is teardown. The
snapshot caller discards only its new clone on failure; retirement frees only its
unpublished entry. The helper allocates and populates a replacement index without
changing old slots or entries, checks its count, locates the candidate slot, then
commits without further fallible work. The existing linked arena owns all graphs.

Lookup returns exact typed-root membership without allocation. Teardown detaches
and frees the index before the existing list/arena destruction; the Environment's
lease preflight still precedes all cleanup. Symbol ownership, nested graph copies,
public escape copies and scheduler/provider lease code are unchanged. I have run
only diff checks at this source checkpoint, not builds, fixtures or timing gates.

## My fixture checkpoint

I extend the existing owning env.c fixture, retaining its full prior suite and
all parsed public escape/scheduler/cache controls. New measured sweeps cover both
snapshot and retirement with 0, 3 and 12 prior roots: initial table, insertion
without growth and the exact first growth. Every observed allocation is failed
in persistent and transient modes. Failure preserves old table bytes, arena head,
entry/graph addresses, aliases, sentinel output and retirement input; recovery
uses the same Environment and then a separate fresh successful control. Successful
growth requires capacity 32 and the same old entry pointers.

Thirty-three actual live empty tuples mapped to sixteen buckets guarantee three
colliding hashes. I retire two, check their exact adjacent probe slots and query
the third as missing; lookup and duplicate refusal run with allocation budget zero.
I retain real string/record/tuple roots, equal text at separate addresses, nested
non-roots and independent Environments. A differently tagged char pointer to a
live tuple's object representation checks the type component without reading it
as a string. No fabricated addresses or invalid memory accesses are used.

Pure size checks cover invalid capacities and product overflow without huge
allocations. A controlled count=SIZE_MAX on an existing valid table checks the
publication overflow refusal before mutation; I restore the count before any
cleanup. All old prefix, graph, list, callable, scheduler, bundle, cache and borrow
staging assertions remain. The two old hardcoded publication loops now include
the required index allocation. No new builds or fixture execution have occurred.
