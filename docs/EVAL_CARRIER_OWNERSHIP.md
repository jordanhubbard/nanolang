# I establish roots before reclaiming carriers

I distinguish three facts in the current1d2e922c48 interpreter:

1. An owning record binding copies its input; replacing the binding releases
   that copy. A borrowed C/API binding keeps the caller's pointer and owns none
   of its storage. My #1207 guards retain this distinction.
2. Both generated typed-list push/get dispatch paths store and return the same
   StructValue pointer, encoded as int64 in an ELEM_INT DynArray. Neither a
   numeric Value tag nor the array's element tag identifies that edge. Generic
   DynArray integer traversal therefore cannot discover this record root.
3. Scope/call cleanup drops aggregate bindings; aggregate constructors and
   returned/manual clones have different copy rules. Missing cleanup leaks,
   while the already-active replacement cleanup invalidates typed-list roots.
   Adding recursive frees at scope exit would add this failure to more paths.

My native API replacement and checked source/shadow observations now establish
fact3 directly. The no-replacement API control passes ASan/UBSan+LSan and
asserts both the input-to-binding copy and binding-to-list pointer identity.
The source failure occurs while copying the value returned by list get into
another ordinary binding. Copying list get cannot repair a header already
freed at replacement; copying on push would change the retained API contract.

My next implementation must account for fresh source allocations independently
of the public Value/container/FFI layouts and legacy caller-created carriers.
It must record the typed-list edge at the actual producer, not guess that an
arbitrary integer is a pointer. The root set must include owned environment
bindings, active evaluation/argument temporaries, returned/host-owned values,
aggregate descendants, typed-list entries, and coroutine/handler lifetimes.
Borrowed roots prevent premature destruction but never grant ownership of
caller-provided storage. Duplicate roots must not imply duplicate destruction.
Popping a frame is not proof that its allocations are unreachable.

I must account for every existing early-discard operation before registering a
carrier: fresh literal retirement, partial-array rollback, array removal,
ordinary binding replacement and environment teardown. Existing raw legacy
cleanup must not leave a stale registry entry or reclaim a registered object
twice. The source-specific constructor/return clone sites must preserve actual
owned versus shared fields rather than pretending create_struct, create_union,
create_tuple and the two manual clones share one layout/lifetime policy.

I do not attach a record-release callback to generic GC shutdown without
checking its order. gc_shutdown currently walks allocation-list nodes while
saving next before a finalizer, and managed STRING child storage can be visited
before a containing malloc-backed record. A new ownership design must handle
both ordinary list release and shutdown, rather than assuming a finalizer has
safe access to every child or may recursively release the saved next object.
This is an implementation constraint from current source, not a claim that an
uninstalled callback currently fails.

I will first require direct ownership/provenance accounting and actual escaped
alias controls, then reclaim unreachable frame carriers, then rerun the full
original compiler/source/platform gates. I retain the1937/82 and3720/159 full
LSan failures. This boundary document and a red witness are not acceptance of
#1211/#1116/#1140 or the release.

## I account for allocations before adapting the evaluator

My private `src/eval/carrier_heap.c` engine owns only allocations explicitly
registered with a destructor and an optional edge tracer. It keeps a private
pointer index and an iterative mark queue; it changes no Value/container/FFI
layout. An unknown root or edge grants no ownership. Duplicate registration,
a missing destructor and mutation/reentrant collection during callbacks are
refused. Allocation failure leaves the payload with its current owner; failed
collection leaves every registered allocation intact. A tracer returning false
reports an incomplete walk and aborts before any destruction; a later complete
walk can retry against the unchanged graph.

My tracers enumerate valid registered edges. Shared nodes and reachable cycles
keep their exact addresses; unreachable cycles retire once without recursive
child destruction. Destructors release only their own carrier/scalar storage,
while registered children have independent entries. An adapter traversing an
unregistered legacy container must inspect its valid edges explicitly; this
engine cannot infer them from arbitrary integers or pointers.

`nl_carrier_heap_forget` transfers one allocation out without destroying it.
My adapter must explicitly transfer every owned descendant crossing a host or
interpreter boundary and remove entries before existing early manual discard.
A transferred graph can be readopted by the same or another heap without a
payload copy. The accounting invariant is registered = destroyed + transferred
+ live. Heap teardown destroys every remaining entry and releases its index.
The engine is per interpreter and does not provide concurrent mutation safety.

My focused normal and ASan/UBSan+LSan tests cover explicit replacement/list
roots, sharing/cycles, caller-owned unknown storage, transfer/early discard,
readoption and cross-heap transfer, reentrant mutation refusals, incomplete-trace rollback/recovery, a65536-node
chain, allocation/removal cohorts and six actual metadata allocation failures.
The graph fixture balances67667 payload objects and their owned text buffers;
the fault fixture separately verifies every registry metadata allocation is
released. These tests exercise the actual private engine, not the interpreter's
source producers. I add their target to ordinary units, but link the engine
only into this focused target until its adapter is ready.

The evaluator/environment/GC paths are still unchanged. Actual source allocation
registration, evaluation/argument roots, typed-list and async lifetime hooks,
host transfers and coordinated early-discard adapters remain required before
collection is enabled. The retained #1211 source UAF and both full-source LSan
failures remain open; an engine unit pass does not accept those source paths.

## I adapt actual carrier storage before source activation

`src/eval/carrier_values.c` now supplies the engine's actual Value-storage
adapter. Adoption takes exactly one record, union, tuple or static-array
header and its owned STRING/FUNCTION scalar storage. Aggregate children are
separate allocations: reaching a borrowed child never adopts it. A source
constructor must register every fresh copied header explicitly; legacy C/API
references keep their existing ownership protocol. Records retain packed or
separate names, and scalar cleanup handles managed and unmanaged STRINGs.
I retain the draft adapter's actual managed-tuple ASan bad-free and qualify
its corrected scalar cleanup against the unchanged witness.

Typed record lists require a producer-supplied descriptor; an ordinary INT
never becomes a root because its bits match a live list pointer. Owned lists
must also be managed GC_TYPE_ARRAY/ELEM_INT containers. Their record pointers
remain shared references rather than copied elements. Explicit extra edge facts
represent typed fields whose public storage contains only an integer. Replacing
those facts is atomic if allocation fails. The source adapter still has to
supply the facts from actual producer/type information; this API grants no new
source or foreign-pointer permission.

An iterative, cycle-aware walk follows registered carriers and valid borrowed
legacy containers. Shared nodes retain identity. Transfer first gathers the
complete reachable graph, then removes registered ownership without changing
payloads; it does not adopt or retire caller containers. Allocation or shape
failure during gathering/marking preserves every owned payload. The private
engine's borrowed context lookup exposes adapter metadata without transferring
ownership. Neither adapter nor engine provides concurrent mutation safety.

My actual-constructor fixture passes with normal source-provider objects,
with the adapter separately compiled at-O2, and with ASan/UBSan plus leak
detection. It checks nested union/record/tuple/STRING/callable/array payloads,
actual duplicate/multiple list roots, borrowed cyclic containers, explicit
field facts, integer refusal, transfer and caller teardown, packed/legacy/
Unicode/empty metadata and self-referential arrays. Five actual creation/
adoption allocation failures, one fact-replacement failure and eighteen
collection/transfer walk failures preserve ownership; the walk failures each
retain129 actual carriers before recovery. All adapter/engine metadata and
owned runtime objects balance at teardown. Earlier link and prototype failures
remain retained; I change no original assertion or compiler/provider deadline.

This adapter is still linked only into its focused ordinary-unit target.
Actual evaluator/environment/GC paths remain unchanged. Source constructor/
clone registration, binding/evaluation/argument/typed-field/host/async roots,
early discard and automatic GC/extern-list lifetime integration remain next.
The original typed-list source UAF and full source leak reports stay open under
#1211/#1116/#1140; these C/API storage controls do not accept the source pipeline.

## I qualify real source allocation tracking before default activation

My private `env_carrier_enable` now connects actual evaluator constructors,
both manual record clones and owned binding copies to a session. It is not
selected by my CLI or ordinary environment creation. A private Environment
pointer stores the session; Value/container/FFI layouts are unchanged. The
tracking-enabled accessor is inline on the disabled evaluator path. Original
programs keep the legacy carrier policy until complete roots are qualified.

I register fresh interpreter-created record/union/tuple/static-array storage
and actual uppercase typed record-list producers explicitly. Deep constructor
record copies are registered individually; manual shallow clones retain shared
children. Legacy external constructor calls are not automatically adopted.
Outside evaluation I retain the old shallow nested-record binding policy.
Tracked binding replacement and literal/callable/empty-union/partial-array
retirement defer to session ownership; unregistered storage retains legacy
cleanup. Teardown retires registered headers before definition metadata.
Recognized host aggregate results transfer out; typed-list get remains a
borrowed identity-preserving result. This staging does not establish complete
host/typed-field/argument/async/external-list or automatic GC lifetime rules.

Both unchanged source fixtures now pass privately enabled ASan/UBSan+LSan:
the original record/union fixture retires19 registrations as18 destructions
and1 caller transfer; the scalar fixture retires18 as16 destructions and2
transfers. All six/seven original shadows complete and caller/constructor
checks remain. The unchanged typed-list source/API witness also passes;
its source run retires all12 registered carriers. No assertion, shadow
selection, static permission or original deadline changes.

I retain the first privately enabled full LSan failures:120bytes/6 allocations
and80/4 are shallow match-binding name/value scratch vectors. My constructor
copies them, so I now release only those vectors immediately afterward. This
also improves the default full reports from3720/159 to3600/153 and1937/82 to
1857/78; default carrier ownership remains incomplete. The source witness
separately exposes379/14 generated-list descriptor allocations. Under #1212
I register only their fourteen explicit blocks with the existing checker
owner. Repeated four-prototype lookup/idempotence/borrowed-declaration and
actual source/AST teardown pass; the unchanged direct control fails against
retained old providers with36632bytes/1456 allocations. AST declarations and
static builtin caches remain borrowed.

Fresh complete normal/sanitizer providers pass all134 original evaluator and88
checker methods (broad sanitizer runs explicitly disable leak detection),
complete engine/storage/binding/borrowed-value controls and generated metadata
controls. Privately enabled source fixtures keep leak detection enabled.
Default source/API replacement still fails ASan; I retain both failures and
the current complete default leak reports. Private qualification is not a
repair of the default source pipeline.

My new actual-source retention observation executes two unchanged64-call
record-case batches. Live registered carriers grow18 ->210 ->402 with no
intermediate destructions. Final teardown balances403 registrations as402
destructions and1 transfer under LSan. This is exact ownership accounting,
not bounded live heap acceptance. I must next supply complete active-value,
binding, checked typed-field/list, argument, host and async roots and external
release/GC-shutdown lifetimes before collecting during calls/loops and enabling
tracking by default. #1116/#1140/#1211 and full source/platform/release acceptance
remain open; I do not replace them with teardown-only success.


## I qualify active roots with a separate private collector

I now keep nested evaluator root scopes. A completed child aggregate remains a
root while sibling expressions run, and newly registered constructor storage
stays rooted until its enclosing scope yields. I collapse child temporaries
on return, retaining only the result in the parent. Lexical statement and loop
boundaries discard consumed temporaries; a while result remains protected
through its final condition because that result is still yielded on exit.
For iteration I preserve the iterable while resetting each body's temporaries.
My ordinary disabled path tests the inline session pointer before root calls.

I retain one checked `record_list` result bit on AST expressions/statements;
I do not infer pointer meaning from an ordinary INT. Live symbol types supply
binding list facts, excluding checker-retained void placeholders. Named record
and union definitions supply current field facts during graph traversal;
explicit tuple-literal element facts follow their checked source children.
These facts create graph edges without adopting their referents. Constructor
and copied-binding registration still supplies ownership independently.

`env_carrier_enable_collection` is a second, explicit, private qualification
step. Ordinary environment creation and my CLI select neither tracking nor
collection. The qualification caller supplies complete, valid host roots;
replacement copies the root descriptors only at quiescence. Host call arguments
remain roots during their activation. This is not an automatic external alias,
list-release or GC-shutdown protocol. Borrowed/untracked carriers are followed
without being adopted. Complete generic/module/array typed-slot provenance,
external call mutations, async roots, low-level constructor capture and GC/list
lifetimes still require qualification before default activation.

I retain two failures from the active-collection candidate. Returning a list
inside a named-record INT field first loses the list edge and produces an ASan
use-after-free. Dynamic definition-derived field facts and explicit tuple facts
repair the unchanged fixture. A later aggregate-producing loop first drops
its preceding result before its final condition and returns freed storage;
keeping that result through the condition repairs it. I also retain the first
bare-variable loop fixture's E034 refusal and replace that fixture expression
with an actual value-producing call, preserving source admission.

My new checked source fixture exercises sibling record calls, tuples, a
256-iteration while and range-for with aggregate-valued bodies, list returns
and record/union/tuple list escapes, with ten actual shadows. The selected
normal/sanitized workload observes a peak of seven live carriers, and two
additional64-call batches finish at seven each. The original complete
callback-record fixture and its six shadows also run unchanged: repeated
record_case calls finish at one live carrier after each64-call batch and
retire403 registrations as402 destructions and one caller transfer. This is
bounded behavior for these actual workloads, not complete #1116 acceptance.
The original scalar seven-shadow and typed-list two-shadow fixtures and
explicit host-list identity controls remain part of qualification.

I preserve the tracking-only growth observation and default source failures.
I do not close #1116/#1140/#1211 or claim current-source paired platform,
bootstrap or release acceptance from this separately enabled collector.


## I retain a native-byte failure beyond carrier counts (#1213)

My selected carrier counters are flat; native allocated bytes are not. A new
external observer retains the complete source fixture and all its assertions,
renames only its C entry point and wraps calls for measurement. Ordinary
glibc `mallinfo2` (allocated arena plus mapped blocks) observes a2,101,904-byte
positive control and36,864bytes of growth over the second64-call batch. ASan's
current-allocation query observes a2,097,152-byte positive control and25,728bytes
of growth. Both original bounded-byte criteria fail with exit94. I preserve
the first observer's C main-macro compile refusal and corrected measurement.

GCStats independently grows by19,584bytes and384objects in that second batch.
A separate linker-wrap/-no-pie diagnostic follows actual STRING allocations
and their final releases. After source context teardown,384 strings from the
final64 calls remain managed, totaling19,584bytes. Three128-object stack groups
identify two AST_STRING contexts and `eval_record_field_value`'s owned STRING
projection. Their384 sixteen-byte hash entries explain the remaining6,144bytes
in the ASan measurement. GC's global lists keep these strings reachable, so
process-end LSan does not report this retained live storage.

I keep #1213, #1116 and full bounded native-heap acceptance open. The next
repair must distinguish explicitly fresh source string references from borrowed
binding/container/C/API references, release consumed temporaries and transfer
binding/return ownership correctly. My private root collector remains disabled
by default. I make no historical timeout or performance attribution.


## I qualify private STRING reference scopes (#1213/#1217)

My private evaluator now registers explicitly fresh STRING constructor and
projection references. Named string helpers that always return fresh storage
also register their results; borrowed identity conversions do not adopt their
inputs. A binding retains a separate reference when its initializer is scoped,
suppressing the legacy alias scan's duplicate retain. Temporary references
remain valid through sibling evaluation and retire when consumed; an escaping
STRING result moves one reference to its parent or caller. My counters count
owned references, not distinct payload allocations.

A managed STRING read now retains a separate scoped reference, preserving its
pointer and its original owner's reference. Multiple read references may name
the same payload. Scope exit releases each consumed reference and preserves
exactly one returned reference. I retain an actual sibling-mutation UAF in both
ordinary/private modes: a later operand replaces a global STRING binding before
the earlier operand uses its value. The private read reference repairs the
unchanged source. The ordinary-mode failure remains under #1217 because private
tracking is still explicit. I do not adopt arbitrary borrowed C/SDK storage.

Both normal and ASan/UBSan+LSan versions of the new ten-shadow source fixture
pass aliasing, owning record rebinding, tuple/union/record projections, helper
calls, loop values, sibling mutation and argument evaluation. Both64-call
batches retain exactly the same93bytes/two objects accounted for by the existing
global STRING slots. Context teardown leaves zero GC bytes/objects and balances
20950 reference registrations as20949 releases and one caller export. The
caller input's pointer/content/refcount remain intact. A separate exact source
loop-return control exercises duplicate references and balances11/10/1/0 with
zero GC objects. I retain the new fixture's first one-global assumption failure
and its two-slot observation; my final assertion derives the baseline from
actual live binding roots, requires no growth in either batch and requires
zero after teardown.

I preserve the first unsupported interior STRING-field mutation refusal and
qualify admitted owning-record rebinding instead. My negative fixture still
requires the original exclusive borrowed-owner diagnostic. Both direct C
library controls preserve caller strings. Their old result owners retired
correctly but leaked13 native scratch bytes in str_concat/str_substring; I now
free only those copied buffers. Two injected scratch allocation failures
preserve caller stack storage and recover. I retain the first fault fixture's
incorrect VOID expectation for substring; its existing empty-string fallback
remains unchanged.

Under #1214 I add an explicit recursive destructor only for newly copied
AST_FOR TypeInfo roots in the existing checker owner. Borrowed AST annotations
and untyped checker blocks keep their existing owners. The completed STRING
fixture first retires every runtime reference and then reports352bytes/two
TypeInfo allocations; the fixed checker-only eight-round metadata control,
borrowed AST read after environment teardown, and32 private runtime shadows
pass with LSan. Default runtime replay still leaks5080bytes/176 unbound-array
allocations under #1116; a separate semantic-only replay completes every shadow.

Under #1215 I fetch a nested static-array element from its existing Value slot;
the original admitted nested-loop shadows had received VOID. The complete
fixture now tests nested iteration, break and continue without a layout or
permission change. Under #1216 I preserve NULL/count/descriptor refusal before
private argument-root observation and pass all134 original evaluator cases in
ordinary and private tracking modes on both profiles. The original array
retirement assertion also passes: its counter now includes actual static-array
frees by the explicit owner, without counting registration or intended cleanup.
Broad sanitizer evaluator/checker runs explicitly disable LSan; focused source,
metadata, ownership and fault checks enable it.

My unchanged native observer now reports zero ASan allocated-byte growth with
its original2MiB positive control. The final ordinary allocator observation
still adds1776bytes over calls65..128 and exits94 against the unchanged1024-byte
criterion. An earlier repaired-provider run adds1584. I retain both and assign
no allocator or product cause. #1213 stays open; I neither raise the allowance
nor claim that an ASan pass accepts the ordinary result.

My CLI and ordinary environments still do not enable the private ownership
session. Trusted helper/foreign construction, asynchronous storage, external
list/GC shutdown lifetimes, complete typed-field roots and full original source/
compiler/platform gates still precede default activation. #1217/#1213/#1211/
#1116/#1140 and whole-release acceptance remain open.

## I qualify STRING scopes independently of aggregate ownership

`env_carrier_enable_strings` now creates only my explicit STRING-reference and
nested-frame session. It does not allocate a carrier graph, adopt aggregate
storage, suppress legacy aggregate cleanup or admit aggregate collection.
`env_carrier_scopes_enabled` identifies this scope protocol;
`env_carrier_enabled` continues to mean actual aggregate ownership. The existing
full-private enable operation still enables both. I can upgrade a STRING-only
session between frames; I refuse an upgrade beneath an active frame, and failed
fresh full-mode construction leaves ordinary mode unchanged.

My [Linux receipt](evidence/string-reference-mode-linux-20261123/README.md)
qualifies the exact sibling-mutation source independently, admitted synchronous
async calls with 130 repeated calls, separate C caller/result ownership and
balanced reference teardown. It also retains the original duplicate-loop-return
source: STRING-only semantics pass, then LSan still finds 59 bytes in four
unbound static-array allocations. Full-private mode is clean. This remains an
aggregate-ownership failure; I do not enable aggregate collection to disguise a
STRING-only failure or disable leak detection for the focused positive units.

All 134 original evaluator cases pass in ordinary, STRING-only and full-private
normal/sanitizer modes. Broad sanitizer suites disable LSan explicitly; focused
units enable it. The current ordinary sibling witness still reports ASan UAF.
Ordinary environment creation/CLI remain unchanged. This separation is a
qualification prerequisite, not default activation or #1217/#1213/full-release
acceptance. The handwritten queued `spawn`/scheduler paths remain unregistered
source operations; my retained parser/typechecker refusals do not establish a
runtime failure on an admitted source route.

## I qualify the STRING candidate through the actual C driver

My [compiler receipt](evidence/string-scope-compiler-linux-20261123/README.md)
links separately named ordinary and STRING-only C compiler images from exact
private providers. I replace only environment construction in the candidate;
production environment creation and five frozen tools stay unchanged. Both
images retain the declared compiler driver and native emitter. All five original
STRING/active-root source fixtures now complete their candidate shadows, publish
native output and execute with their declared main results. The ordinary image
still fails the sibling/async/combined STRING shadows.

This actual route exposed two separate native publication bugs after successful
shadow completion: hardcoded Result payload fields and unmapped generated List
extern names. Under #1218 I use declared Ok/Err fields/tags in the three unwrap
operations, preserving once-only input/default evaluation and panic/refusal
boundaries. Under #1219 I map only operations of registered concrete List
instances, preserving real declarations and local callable priority. Four
native regression methods pass in both ordinary and STRING-only modes; the
unchanged shadow-supervisor suite passes all three methods. My normal and
ASan/UBSan StringBuilder unit suites pass; their sanitizer runs disable LSan.

The candidate completes the first original compiler-source stage in 151.092s
under its unchanged 600-second parent limit and default shadow deadline: all
1306 selected shadows complete without failures, and the resulting native
compiler executes --help. This is not the subsequent self-hosted raw stage,
bootstrap equality, paired platform qualification or whole-release acceptance.
I retain the earlier non-File --emit-nvm refusals and setup failures separately.

A separate experimental record-List loop test is checker-refused before C
emission; #1220 retains that exact red test and the later representation lead.
It does not weaken the four completed native regression methods. General
#1116/#1213/#1217 ownership and ordinary activation remain open.

## I preserve record-list loop metadata and borrowed elements

Under #1220 I resolve List<record> element identity from checked declarations,
parameters, aliases, fields and declared call results. I preserve inferred and
annotated binding names and allocate complete loop-record TypeInfo leaves in
my checker owner. I do not classify ordinary integer bits as record pointers.
The interpreter borrows the existing record-list slot/header, snapshots the
iteration length and keeps the current checked list kind before evaluating a
computed producer. Native lowering calls canonical by-value getters; schema
runtime headers and their storage ABI stay unchanged.

My [Linux receipt](evidence/record-list-loop-linux-20261123/README.md) retains
three successive failures of the exact original witness: checker field refusal,
admitted interpreter refusal, and native name/dereference failure. It then
passes actual shadow/native execution. The complete ten-shadow fixture tests
parameter/local/field/computed/empty iteration, nested loops, early return,
break/continue and list growth. Four contexts per normal and ASan/UBSan+LSan
run retain caller pointer/string identity across 256 source calls and both
explicit caller/environment teardown orders, with zero final GC storage.

I retain the first unit's invalid release-function compile refusal and its
unchecked-shadow collector failure. The latter called my explicit private
collector without the shadow-check phase that supplies inferred List roots.
ASan identifies that missing prerequisite; the corrected unit checks both the
program and its shadows, preserving every source assertion. I do not turn that
setup correction into a claim of safe collection over unchecked ASTs.

Under #1221 I retain the complete accepted native list names in the C driver's
scanner/dedup/diagnostics and emit record-loop type/helper names without fixed
buffers. A valid 66-character name previously passed shadows then aborted at
an unchecked copy into a 64-byte driver buffer. The final seven-method native
suite also qualifies nested distinct record names of 66 and 126 characters
with a common prefix, plus existing declaration/local-callable and prior-output
negatives. Ordinary detection-count policy and runtime/inline selection remain.

The original 134 evaluator / 88 checker cases pass normally and with sanitizers;
broad sanitizer suites disable LSan. The actual instrumented C-driver/native
suite passes seven methods with changed/common providers instrumented and
unchanged backend providers ordinary, again with LSan off. Focused ownership
and checker fixtures keep LSan on. Final source compiles the whole compiler
through the first original C-seed stage in 111.057s under the original 600-second parent and default shadow
limits, with all 1306 selected shadows complete and a working native --help.
The subsequent self-hosted raw/module/bootstrap/platform and default ownership
acceptance gates remain open.

## I observe a precise cache contribution without accepting my ordinary gate

My [bounded debugger receipt](evidence/carrier-tcache-snapshot-linux-20261123/README.md)
reads the unchanged failed native observer at its existing64/128-call allocation
samples. I retain ordinary ASLR, all128 calls, all ten shadows, the original
1024-byte criterion and2MiB positive control. I make no inferior allocation
calls and install no allocator setting. The observed run still exits94, with
+1056bytes. Its thread-cache chunk totals grow11024->12080bytes: bin31 grows
from three to five freed528-byte chunks, accounting exactly for this delta.

Two preceding debugger setup attempts could not read the optimized-out static
TLS symbol and ended before the final sample. I retain those incomplete runs.
The completed reader uses the actual retained AArch64 malloc instructions and
TLS slot, asserts the instruction bytes, and validates every decoded list
length, alignment and chunk size. This is a host-specific diagnostic; I do not
change production allocation policy or treat debugger exit0 as inferior success.
I do not project this run's1056bytes onto the earlier1776-byte observation.
#1213's ordinary acceptance and default ownership remain open.
