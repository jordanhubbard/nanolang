# My bounded nested string-array authority contract

I propose the next required #1001 source-authority slice from aggregate
`4cfad3737`. `workspace_repairs` owns `fix/5.1-nested-string-authority` in the
existing isolated nested-array worktree. This is a design checkpoint only;
no admission change or build is included.

## Existing accepted source and exact boundary

My existing `test_cseed_nested_array_literals.py:NATIVE_REFILL` and
`test_selfhost_array_field_setter.py:test_nested_array_replacement_context`
exercise `struct Box { rows: array<array<string>> }`, empty inner replacement
and refill through a projected alias. Their gates cover C-seed/native/VM and
installed Stage1/Stage2 paths. I have inspected their code, not rerun or claimed
fresh aggregate results. Current ordinary publishers deliberately omit this
metadata while accepting exact depth-two INT. Existing hand-bytecode nested
string slice/GC tests are runtime groundwork, not source authority evidence.

I add only exact `array<array<string>>` fields alongside existing flat and
depth-two INT contracts. I preserve no-authority/UNKNOWN behavior for deeper
nesting, record elements, callable/map/union elements, resources and foreign
bodies. Existing record DAG, direct-tail and ARR_REMOVE boundaries remain.
I do not create a new release requirement from optional LLVM/Wasm refusals.

## Descriptor and normalized proof checkpoint

The original v3 ARRAY_FIELDS type graph remains ARRAY -> STRING -> NO_INDEX,
with complete field bindings and unchanged checked declaration/layout ordinals.
It must remain distinct from ARRAY -> INT even though both outer values have
TAG_ARRAY. I introduce a separate normalized ARRAY_OF_STRING obligation; the
existing ARRAY_OF_INT bit keeps exactly its old meaning. Duplicate equivalent
rows normalize equally; different leaves or depths never collapse.

The current one-byte per-field masks cannot hold an arbitrary new tag bit. My
proposed representation retains that old allocation and introduces a separately
charged optional nested-STRING field bitmap only when new descriptors require
it. A field accessor yields the uint16 normalized obligation, using a distinct
named bit outside the supported flat/INT obligation set. New STRING fields do
not store the INT bit as a surrogate. Existing flat/INT-only work and storage
limits stay byte-for-byte/boundary equivalent; new bitmap/scan allocations have
explicit typed LIMIT/MEMORY tests. I review this representation before edits.

Whole-module monotone proof constrains the outer origin to ARRAY and every
possible child origin to STRING-array storage. I combine surviving children
and all_writes, including detached/replaced aliases and empty outer arrays
refilled later. I impose the exact STRING leaf requirement on each inner origin;
INT/STRING alias collisions, mixed/unknown child origins, wrong depth and
unproved nonempty children refuse. Flat STRING data are string values, not
nested array origins. A single module may contain separate valid INT and STRING
outer graphs; crossing their children must refuse. Calls/results/globals and
accepted acyclic tails preserve these facts at convergence.

Every copied-plan/public verifier, assembly and transport load independently
checks original descriptors and flow. A prior eligibility result is not an
admission token. I retain INVALID for malformed encoding, UNRESOLVED for a
well-formed unproved flow, LIMIT for bounded resources and MEMORY for allocation
failure. Source optional omission versus hard failure remains unchanged.

## String ownership and runtime boundary

Unlike packed inner INT arrays, inner STRING arrays contain boxed string owners.
I retain exact element tags, non-null live string owners and the existing
module-constant/intern-table lifetime. String bytes/length and embedded NULs must
survive array aliasing and module constant lookup; no string-table index may be
reinterpreted as a pointer or assumed stable across copied/loaded modules.
Dynamically concatenated strings are included, so literal immortality cannot
mask a lifetime error. Copied-plan execution must own its retained constants
independently of the caller's original module buffers.

I replace the private validator's boolean INT-only inner mode with an exact
expected leaf fact, independently decoded from the copied original row. Field
acceptance checks both outer ARRAY and expected inner leaf for every existing
child; whole-flow proof constrains future empty-array writes. Private and actual
public execution are separately qualified.

I trace actual retains through LOAD/PUSH/GET, field projection and slicing at
both array levels. Repeated shared child arrays and repeated shared strings
consume cumulative headroom; checks precede mutation and preserve roots on
failure. ARR_SET transfers the popped value under existing semantics and must
not gain a spurious retain. POP transfers ownership; discarded aliases release
exactly once. I inspect the private iterative release bound for record DAG plus
two array layers plus string leaves, rather than assuming the INT bound remains
sufficient. Strings add no back edges; no new cycle admission follows.

Canonical native storage keeps existing NARR nested representation and traced
string/array owners. I require shared inner identity through ordinary record
fields, forward call results, globals, replacement and forced collection at
O0/O2 with ASan/UBSan. Retained string bytes after dropping original outer and
string temporaries are essential assertions. A successful exit alone is not
lifetime evidence.

## Ordered acceptance

1. Review normalized obligation/storage representation and this contract before
   admission edits. Record issue and roadmap ownership.
2. Qualify descriptor/whole-flow proof positives, exact mixed INT/STRING graph
   controls, wrong-leaf/depth/ref/cycle forgeries, detached alias writes and
   unchanged flat/INT work/storage boundaries.
3. Qualify copied private and actual public VM, assembly and decoded transport
   rederivation; verify STRING-vs-INT descriptor tampering refuses and restored
   input recovers. Include source-module buffer independence where the copied
   API promises ownership, and independent module string tables.
4. Qualify canonical native alias/GC behavior, inner string slice/pop/refill,
   duplicate-owner overflow, bounds errors and reached transient/persistent
   allocation failure prefixes with cleanup, prior-result preservation where
   promised, and independent successful recovery.
5. Only then extend both checked source publishers and their exact nested leaf
   serializer, preserving old bytes and unsupported omission. Run paired source
   fixtures including existing empty replacement/refill, aliases, imports,
   concrete generic function instances, forward record DAGs and shadows. Host
   queries classify complete program and shadow bodies independently; no
   unverified staged image escapes.
6. Retain focused local receipts and actual failures. Fresh raw generations,
   exact Linux/Darwin aggregate and full #1001 acceptance remain coordinator-owned
   and open; I start no competing full bootstrap.
