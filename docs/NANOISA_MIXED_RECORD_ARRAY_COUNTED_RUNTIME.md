# My counted mixed record-array runtime acceptance

I base this contract on actual PR924 merge
`ae916a0a821fffdccb9d92b6e62614f97d605801`. MAC
`task_621ca4f76f0344bca70099a0ac9a3934` owns this bounded prerequisite under
f36/15f/488. My [consumer contract](NANOISA_MIXED_RECORD_ARRAY_CONSUMER.md)
retains the complete runtime, public admission and source goal. This checkpoint
contains no implementation or execution.

## Boundary and exact correspondence

I next exercise existing `managed_strings.c`, `managed_module.c` and the runtime
packager. I add fixtures, not an admission selector. My storage core already
accepts counted ARRAY children in records; my descriptor table carries only
`{global_layout_index, field_count}`. It does not enforce an element annotation.

I use an explicit fixture catalog with ordinary record DAGs, all five flat array
element tags (INT, U8, FLOAT, BOOL, STRING), and interleaved unused union layouts.
A host preflight builds the original module, prepares the complete private
record-array report, and compares every relevant layout ordinal/global index,
field offset/type/binding and origin element constraint to independently stated
expected constants. I retain serialized bytes, copied facts and catalog hashes.
My runtime fixtures use matching immutable numeric descriptors and counted
literal bytes. No host pointer or opaque report enters native/Wasm memory; no
trusted report is reusable public authority. I do not execute union values.

A manually written C call sequence checks the storage/root protocol only. It
cannot establish that VM, nvm2c, nvm2llvm, Wasm lowering or source producers emit
that sequence. Those matching consumer gates remain required after this step.
I preserve refusal of nested ARRAY origins in the new query even though my
storage core has broader private graph capabilities.

## Operation and ownership ledger

I define one owner for every live fixture stack/local/global/argument/result or
staging slot. Copying a heap value requires a retain; moving clears its old
slot without release. Borrowed C arguments do not create owners. I never retain
a pointer into my relocatable slot table across allocation.

| Operation | My existing protocol and observation |
| --- | --- |
| Record construction | I borrow a stable vector, validate and retain each child, then publish one record owner. Failure rolls back retained children and leaves the output sentinel unchanged. |
| Record GET | I borrow the receiver and return one retained value. The result survives receiver replacement/release. |
| Record SET | I borrow receiver and new value; retain the replacement before publishing it and releasing the old field. Same-value replacement preserves identity and counts. |
| Array literal/append | I borrow inputs. Scalar lanes preserve their defined packed representation; STRING edges retain counted owners. Growth preserves array handle identity. |
| Array GET/pop | I check retained GET versus transferred pop ownership independently. Empty/out-of-range optional results follow existing VOID behavior; I do not invent a bounds failure for GET. |
| Array SET | My module adapter checks length and reports BOUNDS before mutation. I keep the core's distinct invalid-index status separate; both must preserve existing contents. |
| Array slice/copy | I construct a distinct array, preserve exact elements, and retain STRING children. Mutating one array must not mutate the other. The existing uint32 clamping slice ABI remains unchanged. |
| Call/result/global | I model explicit caller/callee roots, transfer the result before frame cleanup, and retain globals through entry finish until replacement or disposal. |

I collect at documented safe points while inputs remain roots, before beginning
an allocating construction. I never collect halfway through retain/publish/
release. Prepared collection must perform zero project allocations. A failed
collection preparation prevents the operation, preserves its roots, and still
ends an acquired entry exactly once.

My module begin returns status in the low word and acquisition in the high
word. I do not confuse it with finish's result/status packing. An acquired
entry whose preparation fails still requires finish; a BUSY refusal does not
own an entry or inspect replacement descriptor/literal pointers. I preserve
first error across cleanup and refused nested entry. Descriptor/literal storage
remains immutable and alive through disposal; a later different table identity
is refused, not rebound.

## Concrete semantic corpus

For each of the five tags I create distinct arrays A and B. A is owned by a
record field and an outside local; a retained GET creates a third observable
root. I mutate through both aliases, cross capacity growth, replace the field
with B, then mutate/grow A and B independently. I release the receiver and prove
its retained GET remains valid. A full-range slice is a fresh copy whose writes
and growth leave its source unchanged. STRING elements remain valid after the
original string owners are dropped.

I nest the record inside a second record, pass and return it through explicit
fixture functions, store it in a persistent global slot, finish and reenter,
then replace and release the global. Equal-shaped distinct record declarations
must keep different ordinal/global identities. An interleaved unused union
must not shift record ordinals into global layout indices. I cover empty
records/arrays, repeated edges to the same child, and 256 versus 257 module
constructor fields without widening the adapter's 256-field staging limit.

I compare payload bits, not floating equality, for signed zero, finite values,
infinities and chosen NaN payloads through storage/get/copy; no floating
arithmetic is implied. I check INT extrema, U8 endpoints, canonical BOOLs and
counted STRING bytes including embedded NUL and non-ASCII. Signed/extreme
module indices use their actual uint64 representation, and invalid writes
preserve contents and root counts. I keep slicing's uint32 conversion/clamping
observations separate from future source signed-index policy.

I cover successful cleanup, ASSERT and BOUNDS first errors, allocation refusal,
wrong receiver tags, out-of-range fields, null output pointers and sentinel
preservation. For retention failure I use a testing-only saturation setup on a
valid child, restore the counter before cleanup, and prove the old destination
and all unaffected roots remain intact. I do not execute stale/forged pointers.

## Finite fault and memory acceptance

I first measure actual successful allocation positions for each allocating
scenario: array creation/growth/copy, record staging/publication and collection
workspace growth. I sweep each position with one-shot and persistent refusal,
using a fresh runtime/instance and an independent unrestricted recovery after
each refusal. I retain position/count/status, live objects/bytes, expected root
counts and unchanged output/destination checks. Fixture bookkeeping allocations
are distinguished from runtime hooks; libc/sanitizer internals are not counted.

I check exact reference counts only in the instrumented fixture. Separately
linked production tests use observable identity/content/status/lifetime checks.
Retained payload bytes, measured allocator peak and linear-memory pages are
separate counters, never interchangeable claims. I observe peak only if the
fixture wrapper tracks every applicable allocation/free/reallocation path.

I require 1024 sequential create/mutate/replace/finish entries under a 4 MiB
Wasm maximum, with bounded pages after warmup and zero dynamic roots after final
global release/disposal. I include a shorter zero-allocation prepared-collection
sequence with live repeated edges. Existing graph/cycle tests remain unchanged
neighbors; these DAG fixtures do not replace their assertions or admit cycles.

## Routes, tools and retained evidence

I submit complete C/Python/Make fixture changes for review before execution.
The Python module must discover only its intended TestCase; helper modules are
imported as modules. Every command has file-backed stdout/stderr, bounded
TERM/KILL process-group cleanup, explicit status and timeout evidence, including
launch failure. I preserve the first unexpected terminal before diagnosis.

I qualify native GCC/Clang on Linux and Apple/Homebrew Clang on Darwin at O0/O2.
Supported GCC, Linux Clang and Homebrew sanitizer variants retain ASan/UBSan,
leak detection, strict warnings and empty LSAN_OPTIONS. Instrumented core/module
builds and hook-free separately linked builds have distinct names and claims.
I inventory actual compiler commands, Python executable, SDK, linker, LLVM opt,
Wasm engines, sources and provider products before/after each phase.

I use the existing packager to produce and verify native and wasm32 runtime IR,
then link the same observable corpus against those production outputs. I check
its manifest, target/layout, source and IR hashes, absence of `nms_test_` hooks,
and absence of unexpected imports. This is runtime packaging acceptance, not a
new installed public ABI. For Wasm I use import-free modules at O0/O2, `opt`
verification, Wasmtime invocations and Node repeated-entry checks; I retain
module bytes, exports/imports, statuses and counter observations. A missing
required engine/tool is an unmet route, not a silent skip.

My unchanged neighbors include managed records, managed record adapters,
managed array values, record-array origin queries and runtime packaging/schema
consistency as applicable to the actual closure. I retain original arrays,
records, collection and disposal controls; fixture additions do not narrow them.
No bootstrap is implied by these C/runtime tests.

## Required continuation

After reviewed evidence I prepare a separate exact admission/lowering contract:
service-first original-module fresh preparation, complete declaration/origin
agreement, lifetime roots and safe points in every selected VM/native/LLVM/Wasm
consumer, converter/publication preservation, and no fallback after selection.
Generated mixed programs must then agree before and after optimization, including
failure cleanup and package closure. Paired C/Nano producers and all original
shadows follow matching runtime admission. Executable unions, nested arrays,
arrays of exact records and the remaining full mixed graphs retain their own
required extensions. I close neither f36/15f/488 nor full5.1 from this harness.

## Testing instrumentation prerequisite

My existing `NMS_TESTING` budget decrements until persistent exhaustion; it does
not implement a one-shot failure or measure allocation bytes/peak. I keep the
approved distinction and add a separately reviewed testing-only hook at the
real `allocate`/`deallocate` entries. A second explicit macro is valid only with
`NMS_TESTING`. Before each request a fixture callback counts and may refuse it;
after successful backend allocation another records pointer/size; before free
a third removes that exact live allocation. Fixed fixture storage records the
map without recursive allocation. No hook executes in ordinary or packaged
production builds; the existing runtime ABI and budget behavior stay intact.

The fixture owns one-shot/persistent policy, measured calls/live bytes/peak,
map-capacity checks and error reporting. It resets a measurement only when the
map is empty, keeps zero-size successful pointers as live allocations, and
rejects unknown/double frees in its own observation. I review this source
prerequisite before compiling it, then review the complete fixtures separately.

## My fixture checkpoint

I implement three discoverable Python methods: query/catalog correspondence,
native core/adapters, and Wasm/production-package linkage. My query preflight
retains the exact raw CODE/LAYOUTS/OWNERSHIP vectors before destroying the source
builder, then checks copied getter facts against the target descriptors. These
are serialized section bytes, not a publicly admitted `.nvm` artifact. Record
ordinal0/global1 and ordinal1/global2 are equal-shaped ARRAY records;
ordinal2/global3 contains the exact first record; global0 is an unused union.
All five element bindings are checked independently. Separate low-level storage
boundary descriptors do not claim additional source profile admission.

My native and Wasm observable corpus uses all five tags, two distinct counted
STRING roots, exact FLOAT bits, retained GET across receiver destruction, nested
call/result ownership, repeated edges, copy independence and 1024 module
entries with a persistent global. The module corpus rotates element tags and
normal/ASSERT/BOUNDS finish; its finite-memory assertion starts after64 warmup
entries and remains within4MiB. I retain a distinct module preparation failure
where acquisition succeeds, BUSY preserves the first error and finish releases
the active-entry guard before recovery.

The fixed8192-entry observer records every attempted allocation request,
successful backend allocation and free; requested live bytes/peak exclude native
allocator metadata and Wasm block headers. Internal tracker controls cover
zero-byte live objects, duplicate/unknown lifecycle, byte overflow, map capacity
and exact one-shot versus persistent decisions. Seven transaction families
cover creation, growth, slice, record creation, collection preparation,
record/slot/prepared-workspace growth and array literal construction. Each
measured refusal has its own fresh unrestricted recovery. Failed operations
preserve live objects/bytes, input references/content and output sentinels;
cleanup remains under persistent exhaustion. Testing-only saturation checks
both SET preservation and partial constructor-retain rollback. Production IR
contains no hook references.

I use fixture-only volatile memory helper loops for compiler-generated O0 Wasm
aggregate copies/initialization; I still require zero module imports. The
existing production packager remains O2; I test linking its exact IR with fixture
O0/O2 separately from direct runtime O0/O2 builds. I replace only the packager's
command transport and temporary-directory retention in the Python harness, so
failed tool output and produced bytes survive. Its real flags, source hashes,
IR verification and package format remain unchanged.

My execution plan selects the three new methods explicitly and retains the
unchanged managed-record, record-adapter, managed-array-values and origin-query
neighbors as separate phases. Existing adapter controls retain descriptor-table
rebinding, wrong receiver/output sentinels and256/257-field checks; new core
controls add repeated edges and retained rollback. Seven compiler/sanitizer
configurations cover native/query methods; explicit ordinary LLVM phases on
both hosts cover the complete Wasm/package method. I do not repeat Wasm once per
unrelated native compiler or call that skipped acceptance. Exact prospective
phase/tool commands remain a qualification-driver checkpoint. No build or
fixture execution has run at this source checkpoint.

Instrumentation claims are route-specific: direct observed runtime builds
instrument the included core/module; query preflight links the selected ordinary
query provider objects; production package IR remains the packager's ordinary
O2 output even when its calling fixture is sanitizer-instrumented. I inventory
each closure and do not label all package providers sanitized. My unchanged
array neighbors are `test_managed_array_copy_runtime`,
`test_managed_array_graphs` and `test_managed_string_arrays`; I keep the existing
record and adapter suites as well. Explicit numbered method names put copied
query correspondence before target execution in the default new suite.
