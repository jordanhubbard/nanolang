# My map constructor parity checkpoint

I track the complete outcome in #1083, under #1081 and #976. My original
`tests/unit/test_map_constructor_contexts.nano` remains unchanged. I retain
all four int/string key/value pairs, every original context, enumeration
snapshot and clear assertion.

My checker currently returns an unknown named `HashMap` for the empty builtin.
A nested generic-union payload therefore loses its concrete expected type.
I refine only a zero-argument identifier call to `map_new`, with an exact
supported scalar-pair expected type and no real same-named nominal declaration.
I leave general unknown-type equality unchanged. My mandatory shadows include
all four concrete payload pairs and wrong-pair, unknown-value, wrong-arity and
same-named nominal controls. These shadows still require fresh execution.

I expand the existing Python matrix to require the seed NanoISA producer and
both fresh installed stages. Each emits a distinct verified module, executes it
in my VM, translates it to C and executes the result under ASan/UBSan. I retain
the original C-native seed counterpart and expand the existing refusal controls
to both stages. The Make target now requires these actual providers. My hosted
diagnostic workflow runs this complete matrix after fresh bootstrap even if a
focused string test fails, then still collects the complete original JUnit routes.

My local actual-lexer check accepts all 12 static Python fixture fragments;
Python AST, workflow YAML and diff-whitespace checks pass. These checks do not
establish compiler or native parity. Local disk pressure holds substantial builds.

My producer still needs all four typed key/value pairs, keys/values snapshots
and once-only clear. My C AOT analysis/runtime still needs corresponding
representation, alias, lifetime and allocation-refusal coverage. I keep #1083
and both roadmap rows open. This checker foundation does not complete the map
outcome, ordinary-record publication, native adoption, or release acceptance.

## My producer implementation pending execution

I now carry each of the four supported key/value pairs through my self-hosted
map type helpers, exact HM_NEW operands, key/value checks and inferred
keys/values array types. I refuse uncontextualized construction rather than
silently selecting STRING-to-INT. My map clear lowering saves the receiver once,
takes a detached key snapshot and deletes each key in reverse snapshot order.
It emits existing VM operations and leaves no result on the statement stack.

I add mandatory shadows for all constructor operand pairs, typed enumeration,
clear lowering, missing context and wrong arguments. I add runtime controls for
all four pairs: snapshots survive replacement/deletion, empty snapshots grow,
clear preserves aliases and evaluates a returned receiver once, maps can be
reused, and integer keys retain signed extremes. I retain every original
constructor assertion and seed-native counterpart. My Python actual-lexer check
accepts 20 static fragments. These source checks do not establish executed
producer semantics; the fresh hosted run and the C AOT/runtime work remain open.

## My exact scalar runtime storage

My standalone C runtime now stores INT keys as full signed integers and STRING
keys as copied text in a tagged map family. A key union retains the prior
entry storage width on this 64-bit host. I preserve the original string-key
API as checked wrappers. My typed operations retain exact key/value tags,
replacement/deletion aliases, copied result ownership and byte accounting.

My actual runtime fragment passes four standalone ASan/UBSan methods locally:
the complete unchanged original string-map runtime fixture (4,096 entries and
32 forced hash collisions); all four pairs with 1,024-entry growth, signed extrema, copied inputs/results and
zero live allocations after successful teardown; nine invalid tag/null/overflow
boundaries; and every allocation-failure prefix in a smaller complete lifecycle
(15 INT/INT, 41 INT/STRING, 27 STRING/INT and 53 STRING/STRING). My independent
allocation probe counts actual live pointers/bytes. For refusal controls only,
my test invariant hook exits 70 immediately instead of the production abort;
I require its exact diagnostic and reject sanitizer diagnostics. I make no
failure-recovery or failure-path cleanup guarantee from a terminal refusal.

My small dedicated workflow repeats these actual fragment controls with GCC and
Clang on Linux and the native compiler on macOS 15, retaining compiler/input
identity and logs. My ordinary unit target includes them. AOT analysis/emission,
keys/values snapshot storage and source-native end-to-end acceptance remain open;
this runtime storage checkpoint does not admit the complete map outcome.

My [local runtime log](runtime-local.log) records `make -f Makefile.gnu
test-map-scalar-runtime`: four methods pass in 2.637 seconds with Apple clang
21.0.0. My [input hashes](runtime-local-inputs.json) identify the actual tested
fragments and original fixture; this is local runtime evidence, not a fresh
source compiler product or a verifier-owned native lifecycle receipt.

## My failed checker bootstrap and corrected fixture

Fresh runs [42417505641](https://github.com/jordanhubbard/nanolang/actions/runs/42417505641)
and [42417814646](https://github.com/jordanhubbard/nanolang/actions/runs/42417814646)
pass their default fast/report/storage gates, then fail during the seed's
`check_empty_map_context` shadow. Neither reaches Stage 1, the original map
matrix or complete JUnit. My [original logs, manifests, jobs and artifact hashes](bootstrap-failures/artifacts.json)
retain the failure. Both exact seed logs report `parser_get_call` returning void
at the new nominal-map assertion. I do not classify this as infrastructure.

My new fixture used `struct HashMap<K,V>` although my self-hosted struct parser
requires an opening brace immediately after the name. It therefore registered
neither the nominal declaration nor a call. I correct only this new premise to
a real `struct HashMap`, assert its parsed declaration/call and parser validity,
and retain the same requirement that I do not refine that nominal name to the
builtin map family. I also refuse parser-error and out-of-range call inputs
before list access, with mandatory controls. My original constructor fixture
and assertions remain unchanged. Fresh execution of this correction is pending.

## My C AOT integration pending fresh execution

I now construct exact INT/STRING key shapes, check optional key/value uses
without rewriting their producer storage, and emit typed runtime operations.
My STRING-key primitive paths retain their existing API. Optional keys retain
explicit checked string/integer extraction, and my private copied-proof profile
continues to require its original STRING-key shape; this change does not widen
that admission boundary.

I derive keys/values array storage from solved element shapes and refuse
unresolved snapshot types. Each snapshot owns a new array; STRING elements are
copied into my existing owned string-array registry. I check element counts and
retain my existing array allocation accounting, tracing and cleanup. New raw
VM/C AOT controls exercise all four pairs, empty-array growth, optional writes,
replacement/deletion, signed extreme keys, detached snapshots across collection,
forward map parameters/results and prior-output refusal. I retain the complete
original source matrix and existing map lifetime/storage controls. I update the
old C unit's unsupported-constructor premise from newly supported INT/INT to
BOOL/INT, with explicit new all-four-pair positive controls.

My local strict C99 syntax check passes without writing an object. This is
syntax evidence only, not full compiler/native/profile acceptance. My dedicated
fresh hosted GCC/Clang diagnosis builds the actual VM, assembler and translator,
then executes the new cases plus unchanged storage/lifetime/debt/optional-array
modules. Snapshot allocation-refusal controls and complete source/fresh-stage,
Darwin, integrated CI and native lifecycle acceptance still require completion.

My [AOT preparation controls](aot-preparation-inputs.json) retain the strict syntax
command and exact hashes. My [first standalone check](aot-preparation.log) catches
an unused new helper in the original strict runtime fixture; I route the
existing key validation through that helper and preserve the warning policy.
The [corrected runtime check](aot-preparation-corrected.log) passes four methods
in 3.112 seconds, now including eleven invalid runtime boundaries. I retain
both attempts. This does not establish that the changed AOT translator executes
correctly; its new raw-module and unchanged regression gates remain pending.

My corrected-fixture run [42418705413](https://github.com/jordanhubbard/nanolang/actions/runs/42418705413)
then refuses a new shadow's call to private `parser_with_error` during C-seed
source checking, before shadow execution. I retain its full logs, manifest and
exact job under `bootstrap-failures/`. I replace that test-only private call
with an actual invalid source parse and assert the resulting parser-error flag.
I preserve both valid nominal-declaration and invalid-node/error refusal
requirements. I do not widen parser exports or remove any assertion. Fresh
execution of this concrete correction remains required.

My [complete hosted runtime and AOT checkpoints](hosted/README.md) retain the
three-platform/family standalone runtime run and both 36-method fresh Linux
VM/C AOT runs, including full logs, exact source/input identity, unchanged
providers and zero-skip reports. Generated snapshot allocation refusal and
complete source/fresh-stage/Darwin/integration acceptance remain open.

## My generated snapshot denial controls pending execution

I add exhaustive allocation-denial controls around actual fresh translator output
for all four scalar pairs. I retain each original assembly/module/generated C
program, force my existing collection safepoint, and compile native probes at
O0 and O2 under ASan/UBSan. An independent allocation oracle requires zero live
pointers/bytes after a successful complete lifecycle. I measure each actual
allocation count, deny every position in transient and persistent modes,
require the production native-invariant SIGABRT and exact diagnostic, reject
sanitizer failures, and require a clean fresh-process retry with the same count.

My probes include empty snapshot growth, populated key/value snapshots and
snapshot reads after map deletion. They retain every command, status, stdout,
stderr and generated input under unique `.test_output/scalar-map-aot-prefixes/`
roots; hosted artifacts include those roots even on failure. Python AST and
workflow YAML checks pass locally. The changed translator/source pipeline and
these allocation sweeps require fresh hosted execution; their presence is not
acceptance. The same checkpoint also repeats the latest standalone runtime on
Linux GCC/Clang and macOS, since the checked key-kind helper postdates the first
passing Darwin runtime revision.

## My expression-boundary correction

My [three exact context bootstrap failures](context-bootstrap-failures/artifacts.json)
reach the mandatory four-pair constructor shadow and fail its positive assertion.
I retain their full seed/bootstrap logs, manifests and exact jobs. My expression
checker computes the genuine empty constructor's named unknown `HashMap`, then
falls through to an unresolved-function path which relabels it `map_new`.
The contextual helper therefore cannot recognize its declared sentinel.

I preserve that sentinel only for an unbound empty-map builtin, check zero
arguments explicitly and keep bound functions on their existing symbol path.
New mandatory expression shadows check the exact sentinel, real declaration
identity and wrong arity. The original four-pair constructor assertion remains;
I add parser-validity and diagnostic reporting before it so any next failure
identifies the actual typecheck reason. Fresh complete execution remains required.
