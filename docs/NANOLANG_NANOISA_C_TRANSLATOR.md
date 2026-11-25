# My NanoLang NanoISA-to-C translator

I track this implementation under [#1142](https://github.com/jordanhubbard/nanolang/issues/1142).
My release path is source → NanoISA → NanoVM, or source → NanoISA → C emitted
by NanoLang code → host C compiler → standalone native binary. I keep the
[complete release scope](RELEASE_5.1_SCOPE.md).

## My implementation boundary

My translator owns decoding, admission, representation analysis and emission in
NanoLang. File reads, output publication and the resulting program's declared
runtime services may use explicit host interfaces. Calling the C translator,
asking it for a per-program execution plan, or embedding a VM does not implement
this translator. Static C runtime support remains C: it is the target language,
not a substitute for NanoLang analysis and emission.

I retain the C translator as my reference and bootstrap tool until the
replacement translates the complete required corpus and compiler products.
`src_nano/nvm2c/` is the new implementation. I do not change product selection
on the strength of one component's tests.

## My current responsibilities and replacement order

| Order | Current authority | Responsibility I must implement in NanoLang |
| --- | --- | --- |
| 1 | `nvm_format_v2.c`, `nvm_format_v2.h` | Read the 40-byte v2 header and 24-byte directory entries; preserve exact version, feature, flags, range, duplicate, overlap and checksum decisions. |
| 2 | `nvm_v2_cursor.c`, `nvm_v2_constants.c`, `nvm_v2_signatures.c`, `nvm_v2_layouts.c`, `nvm_v2_functions.c`, `nvm_v2_imports.c` | Decode bounded payload tables, explicit byte lengths, exact tags, signatures, function spans, imports and reserved fields. Keep embedded zero bytes and nominal identity. |
| 3 | `nvm_v2_module.c`, `nvm_v2_convert.c`, `modules/nanoisa/nanoisa.c` | Preserve section-presence/default-empty behavior, cross-table indices, entry point, feature/section agreement and declared operand depth. The current bridge to the old in-memory module is an implementation detail I must not mistake for the wire format. |
| 4 | `isa.c`, `verifier.c`, `verifier_types.c`, ownership/reference/layout/service contracts | Decode each instruction exactly; validate branch targets, stack effects/joins, calls, locals, ownership, borrowed places and declared service authority before executable admission. Transporting a section is not admission. |
| 5 | `nvm2c_shape.c`, `nvm2c_callables.c`, `nvm2c_reachability.h` | Derive stable representation constraints, producer tags, aggregate projections, indirect target sets and reachable bodies. Preserve failure poisoning and refuse unresolved/conflicting facts. |
| 6 | `nvm2c.c` and its included emission/runtime fragments | Emit prototypes, C operators, locals, simultaneous edge transfers, direct/indirect calls, tail restarts, strings/byte strings, aggregates, collections, globals, declared host imports and checked lifetime cleanup. Preserve signed endpoints, eager order and once-only effects. |
| 7 | File/Socket/WebSocket/service profile adapters and admission plans | Reconstruct every required admitted profile with its existing grants and lifetime checks. Private diagnostic profiles do not become public merely because I can emit them. |
| 8 | `nvm2c_main.c`, `file_cli.c`, `src_nano/nanoc_v06.nano`, bootstrap/install recipes | Parse the full required CLI, preserve source identity and prior output on refusal, publish checked C, link the declared host closure and install the actual NanoLang-derived translator. |

Paths without a directory prefix in this table are under `src/nanoisa/`.
This inventory maps responsibilities. It does not claim that the C headers'
introductory subset prose enumerates every later admitted operation; actual
admission branches, code, fixtures and release rows remain authoritative.

## My component compiler products

My route harness accepts one explicit source compiler product:
`NANOLANG_TRANSLATOR_SOURCE_COMPILER` names an actual native executable, or
`NANOLANG_TRANSLATOR_SOURCE_MODULE` names an actual raw compiler module that I
invoke directly in NanoVM. A verified bootstrap seed is a development producer;
it does not establish Stage1 generation, native compiler qualification or raw
Stage1/Stage2 equality. I record its source, host closure and tool identities
separately. I create no wrapper or stage alias.

My authoritative NanoISA bootstrap already gives the complete compiler suite a
separate thirty-second shadow deadline and each build stage eighteen hundred
seconds. Its receipt records both. Ordinary compiler invocations retain their
ten-second default. A failed ordinary creation remains failed when a distinct
bootstrap workflow succeeds.

## My container contract

`src_nano/nvm2c/container.nano` is a pure NanoLang implementation of order 1.
Its `validate` result matches `NvmV2Result` at the container boundary. I compare
with `nvm_v2_validate`, not whole-module deserialization, because these are
different contracts.

I validate header fields in the C reference's order, then every directory entry,
then pairwise overlap, then CRC32 over bytes after the header. An empty section
can name a location in the header; its range must still fit the file. I preserve
that current decision. Unknown ISA versions and missing payload sections are
not accepted for execution by this function: their checks belong to module and
instruction admission, which is not yet implemented here.

My language's `int` is signed 64-bit. I decode a wire `uint64` offset only when it
fits the nonnegative addressable range. A set top bit produces the internal -1
sentinel and the same range/total-size refusal as the reference for any
addressable array. I check subtraction bounds before reading; I do not add
untrusted offset and size. My CRC32 uses nonnegative integer arithmetic and no
foreign checksum/decoder helper.

`tests/nvm2c_container_driver.nano` supplies real file bytes to the same library.
`tests/nvm2c_container_reference.c` exposes the existing C boundary without
reimplementing it. `tests/test_nanolang_translator_container.py` builds wire bytes
independently with Python `struct`/`zlib`, checks named expected decisions and
compares all decisions through native, VM and C AOT execution. My default test
runs three routes; setting `NANOLANG_TRANSLATOR_SOURCE_COMPILER` to an existing
NanoLang compiler driver additionally requires that producer through VM and
sanitized C AOT. The inspected driver is built from `src_nano/nanoc_v06.nano`;
the single-file `nanoisa_emit.nano` tool does not merge imported modules. I run
all five routes for this initial component. A valid envelope
with no payload sections is a container test, not a runnable program.

I retain the initial driver fixture refusals (undeclared CLI operations, then
unsafe expression placement) and the raw single-file emitter's expected imported-
module refusal in my component evidence. I correct the fixture by declaring CLI
operations inside an ordinary `unsafe` statement block and use the full module
driver. I do not change a compiler admission rule to make this fixture pass.

## My core payload tables

`src_nano/nvm2c/payloads.nano` implements the ten public core table codecs:
metadata, constants, signatures, layouts, functions, globals, imports, links,
debug and callbacks. `PayloadTable` holds validated row spans into unchanged
section bytes. It preserves full unsigned function/debug fields without
converting their high bits into a signed offset. I apply code-range admission
later, against the CODE section. The translator must retain those section
buffers and leave their bytes unchanged while using these internal views.

`reencode` writes the count and validated rows, omitting tolerated trailing
bytes. My C oracle decodes actual fields and calls the original encoder; my
NanoLang view must reproduce those bytes, including embedded zero payloads and
high-bit fields. Error results publish no partial row list. Padding errors,
unknown tags and reserved fields preserve the existing precedence. Callbacks
require exact section size and sorted unique import/parameter pairs. Ordinary
core codecs retain the current trailing-byte tolerance, including empty tables.

My layout decoder preserves prior-only transport and the public forward-record
profile. Forward edges trigger a whole-table scalar/record shape check followed
by iterative cycle detection; I do not use recursive traversal or accept a
private array/map/ownership profile through this public entry point. Structural
layout decoding is not execution authority.

`tests/nvm2c_payload_reference.c` compares the actual C-decoded/re-encoded fields.
My independent fixtures cover all sixteen constant/signature tags, binary zeros,
full unsigned endpoints, every truncation of representative tables, padding and
refusal precedence, callback ordering, and shared/deep/cyclic record graphs.
`tests/nanolang_translator_routes.py` shares the route build/link machinery with
my existing container gate. Both original three-route commands and the explicit
five-route source-compiler setting remain available.

Passive, ownership and service extension bytes, module cross-table checks,
callback ABI/host semantics, instruction verification and execution profiles
remain separate required work. The current whole-module decoder can represent
absent sections as empty tables; I do not invent a blanket required-section set
from the serializer's normal output. This payload layer changes no product
selection or host grant.

## My core module view

`src_nano/nvm2c/module.nano` combines the core tables and preserves their
cross-section contracts: indices and optional sentinels, unsigned code ranges,
absolute binary-safe artifact paths, callback signature/adapter/coverage rules,
entry points and feature floors. I preserve default-empty sections. Metadata
success still does not verify instructions or grant execution authority.

I derive primary instruction widths from the same `spec/nanoisa.yaml` generator
as my C metadata. My File scan follows the existing C classifier: I skip operand
bytes, stop at undecodable input and recognize even a truncated File opcode at
an instruction boundary. I do not silently treat that classifier as a verifier
for the extended instruction plane.

`decode_core` reports -1 for still-unimplemented service sections and private
array/map profiles. Shared ownership conversion is implemented below. That result is distinct from zero and
from C format errors. It is a development boundary, not a final translator
refusal policy. I must implement those profiles before product cutover; this
module does not change executable selection.

My module tests retain actual whole-module C decoder decisions, existing
unmodified language examples from both producers, all256 primary widths and
all256 scan controls. Qualified forwarding also retains a wrong-nominal argument
and prior-output refusal across producers. These tests exposed two prerequisites:
[#1145](https://github.com/jordanhubbard/nanolang/issues/1145) binds annotations in
the declared module context, and [#1146](https://github.com/jordanhubbard/nanolang/issues/1146)
checks record arguments before either frontend can publish bytecode. I preserve
the qualified source and distinct declaration identities while repairing them.

## My instruction decoder

`src_nano/nvm2c/instructions.nano` decodes the existing primary instruction plane
using schema-generated widths, operand kinds and stack effects. I check the
whole requested byte window by subtraction before reading. A zero length means
refusal; I return no partial operands. Variable stack effects remain -1 for the
later call/signature verifier to resolve.

I retain each operand as unsigned low/high 32-bit halves. `integer` reconstructs
signed I32/I64 values without overflowing at either signed extreme. F64 operands
remain raw bits: I preserve signed zero, infinities and quiet/signaling NaN
payloads without numeric conversion. `reencode` reconstructs the bytes from the
decoded fields. Its input contract is an `Instruction` returned by `decode`.
Neither decoding nor re-encoding grants executable admission. The reserved 0xff
prefix remains undecodable by this primary API, matching the existing C reader;
I do not claim extended-plane verification.

My instruction gate compares actual `isa_decode` operands, `isa_get_info` stack
metadata and `isa_encode` bytes. I cover all256 primary allocations, every
truncation, exact and offset windows, all-ones/high-bit/random patterns and
independent signed-integer/F64-bit controls. My [paired instruction evidence](evidence/nanolang-instructions-20261120/README.md)
retains3472 instruction cases per route within the combined5486-case component
gate on Linux/Darwin. Passive CFG/closed-call checks, extension profiles and
executable selection remain separate work. My initial
fixtures expose [#1147](https://github.com/jordanhubbard/nanolang/issues/1147)
(caller locals hide callee globals during C shadow execution) and
[#1148](https://github.com/jordanhubbard/nanolang/issues/1148)
(accepted INT/U8 joins fail native integer-array shape constraints). I retain
both unchanged witnesses. Namespaced metadata and explicit integer conversion
let this decoder continue; they do not close those release defects.

## My checked global-binding prerequisite

My decoder shadows exposed #1147: a caller local hid the callee's global when
my C interpreter repeated name lookup. I now preserve the checked declaration
identity on global reads, writes and callable selection. Evaluated globals retain
that identity, and qualified imports read current storage. I retain local lookup,
callee-before-argument order and independent callable copies without changing
frame cleanup. My [paired global-binding evidence](evidence/global-bindings-20261120/README.md)
keeps functional/ownership results separate from that checkpoint's legacy
qualified-global AST-to-C refusal (#986) and full original LeakSanitizer failure
(#1149).

My subsequent [native storage checkpoint](evidence/native-global-storage-20261120/README.md)
restores the unchanged qualified fixture's fifth execution route. Typed public
storage accessors preserve module identity, aggregate types, once-only ordered
initialization and callee-before-argument order across separate C objects.
Canonical object and metadata symbols distinguish same-basename modules. My
full NanoLang driver also matches the C loader's duplicate-public-identity
refusal before publication. These prerequisites do not implement translator
admission or C emission; #986 integration and the complete #1149 gate remain open.

## My passive closed-call prerequisite

`src_nano/nvm2c/passive_calls.nano` analyzes concrete local callees under the
assumption that their actual arguments are scalar. It does not infer that fact
from declaration tags. I decode the whole function, follow reachable direct and
tail calls, check stack-height joins, and intersect definitely initialized local
sets until the worklist converges. I check loads after convergence, including
all loop predecessors. A returning path must leave the declared result count.
I refuse recursion, captures, reachable foreign/indirect calls and effects.
Unreachable decoded instructions do not contribute to this summary; ordinary
module verification still applies to them.

I preserve the existing limits:64 active calls,65536 decoded instructions,
2097152 local-state words and1048576 worklist steps per callee. I represent a
64-bit local set word as two nonnegative integer halves and implement set
intersection in NanoLang. The storage limit counts the original64-bit words;
my two-half representation uses twice as many integer slots. I call no foreign
analysis or bitset helper. The API takes explicit narrow function descriptors;
it is not a wire decoder or an executable admission entry point.

`tests/nvm2c_passive_calls_reference.c` includes the existing C implementation
to expose its private analyzer and allowlist, without copying that algorithm.
My differential gate covers both512 opcode/profile decisions and complete
callee decisions. Explicit controls exercise actual depth/instruction/storage
limits and a cyclic set-propagation graph exceeding the actual worklist limit;
I do not lower a limit for testing. Independent graph propagation counts the
stress fixture's steps. This prerequisite alone does not admit passive records.
Guarded inputs, node/block ranges, graph order and whole-module relationships
remain separate checks.

My [paired closed-call evidence](evidence/nanolang-passive-calls-20261120/README.md)
matches1498 callee decisions and512 opcode/profile decisions per route on
Linux/Darwin. It includes415 explicit callee controls and independent actual-C
ASan/UBSan runs, without changing a production bound or the active C verifier.

## My complete passive-claim validator

`src_nano/nvm2c/passive.nano` implements the existing `nvm_passive_valid`
boundary in NanoLang. I validate both record versions, bounded fields and
sorted exact dependency/read lists, guarded immutable parameter inputs,
distinct result locals, stable lowest-source-index topological scheduling,
and exact physical partitions. Version 2 additionally checks forward-only node
CFGs, instruction boundaries, matching stack heights and its unchanged 65536
instruction limit. Version 1 retains its existing straight-line behavior.
I scan the owner function for interval-entry and external-input-write rules.

My `closed_sequence` API checks the encountered concrete callees in block,
selected-node and instruction order using one shared cache. I collect those
roots while checking the claim and analyze them before returning success. The
C boundary returns a boolean; I publish no intermediate claim or execution
grant. Sharing its cache preserves decisions at the analysis depth bound
without exposing caller-forged checked-state arrays.

This API takes bounded code and narrow function/parameter descriptors; it does
not replace v2 wire conversion or ordinary verification. In particular, owner
calls outside node ranges retain ordinary target/stack checks, and module
conversion still owns signatures, references, ownership and other profiles.
`decode_core` now checks the complete core/passive conversion path before
publishing a module view. My active verifier and product selection remain
unchanged.

My [paired passive-record evidence](evidence/nanolang-passive-records-20261120/README.md)
matches 2183 claim decisions per route on Linux/Darwin, including 414 explicit
expectations, four unchanged producer products, the full earlier call-analysis
regression, actual-C sanitizer checks and all25 original passive methods.

## My passive module-conversion boundary

When section 12 is present I preserve its exact bytes and framing/feature
checks, then apply the current C conversion requirements before passive-claim
validation. All constants must be strings with distinct length-and-byte
identities, including embedded zeros. Code, function spans and debug offsets
must fit the narrow carrier; function result counts must fit its byte field.
I reconstruct parameter tags and retained-layout size in conversion order.
Core references, callback shapes and layout DAGs have already been checked.
Ordinary import construction adds no result-count restriction beyond the
existing wire contract; I preserve that distinction from function conversion.

These conversion checks run for a present passive, ownership or shared service
payload. A core module without extension payloads retains its existing
nonconversion decisions. Shared service decoding now follows the boundary
below; private ARRAY/MAP profiles retain their quantified pending decisions.
Successful deserialization does not establish instruction/host execution authority.

My [paired passive-module evidence](evidence/nanolang-passive-modules-20261120/README.md)
matches 2240 complete module decisions and exact claim bytes per route on
Linux/Darwin, including 472 explicit expectations and four unchanged producer
modules. The updated 5489-case decoder gate and prior passive/call gates pass;
four profiles remain pending. I retain the missing-PyYAML setup failure and
the corrected system-Python schema result separately.

## My reusable conversion facts

`module.convert` accepts my decoded `ModuleView` and returns a `Conversion`.
I reuse the existing constant identity, narrow-carrier, layout, routing,
service, ownership-declaration and passive checks in their existing order.
I expose the checked code bytes, exact string bytes, canonical retained
layouts, layout-kind counts, function spans and parameter/result projections
through its `owned.Module` model. A refusal retains its status and returns an
empty model. My existing `decode_core` decision order stays unchanged.

A caller may request conversion even when an ordinary module has no extension
sections. This catches conversion-only refusals that descriptive core decoding
intentionally does not perform. Conversion still accepts, for example, an empty
result-bearing body or an unknown opcode when no declaration checks its body.
I require separate complete structural and instruction verification before
execution. I preserve `-1` for my unimplemented private profiles.

`make test-nanolang-translator-conversion` compares every projected fact with
my actual C reverse converter. I independently classify private ownership
routes with the existing C routing oracle, retain their C refusals separately,
and require identical explicit deferral across my selected product routes.
My [paired ad2 checkpoint](evidence/conversion-projection-darwin-20261121/README.md)
passes6258 complete reports,105 independent expectations and six actual source
products through five routes per host. All3560 guarded inputs, manual carriers
and complete reports match Linux/Darwin and their exact Git blobs. Targeted
actual C projection/reverse-converter sanitizers and the original module methods
pass; I retain the first observer/setup failures separately. This component
grants no full-module, affine, private-profile or pure-emitter acceptance.

## My ordinary whole-module verifier

`src_nano/nvm2c/module_verifier.nano` composes my checked module view and
conversion with the ordinary function-body verifier. `verify` preserves the
loader's positive declared-depth checks before the complete ordinary unlinked
module pass. I check every function's metadata before any function body,
including the empty-result rule, local/signature consistency and overlapping
nonempty spans. I preserve import limits and service-policy priority.

`verify_declared` checks only the declared-depth obligation. With no positive
declaration, I preserve the C API's absence of that obligation: success does
not imply that malformed bodies or other structural errors were checked.
Accordingly, its successful result has `complete=false`. A successful `verify`
has `complete=true` only for the ordinary unlinked pipeline. I keep transport
status, exact ordinary refusals, known malformed-body refusals and explicit
affine/ownership/private-profile deferrals distinct. Refusals expose no partial
depth proof. Inert ownership metadata with no functions and no required walk
follows the C module's ordinary path.

My C oracle calls the actual public declared-depth, full-module and maximum-depth
APIs after actual v2 deserialization/conversion. I normalize malformed-body and
affine refusals separately and retain their original public diagnostics. My
fixtures include whole-module metadata/body priority, selected/unselected
functions, zero-depth absence, exact bounds, overlapping/adjacent/empty spans,
multiple-function bodies and materialized ordinary body contexts. Direct-helper
contexts that cannot be represented by that module carrier, require a linked
graph, or would require large artificial pools remain explicitly counted apart;
I do not relabel the older helper corpus as whole-module coverage.

My [Linux checkpoint](evidence/ordinary-module-verifier-linux-20261121/README.md)
passes7132 modules in both modes through five routes,
126/122 independent expectations and six actual source products per mode in
1444.113s. The actual VM agrees on131 independent consumer controls. Targeted
actual public C verifier/type/decoder/ISA/module/reverse-converter sanitizers
match both7132-case modes. I preserve the initial four over-deferred inert
ownership modules and their precise no-function/no-walk correction.

Paired Darwin qualification remains pending during Puck's reserved capture run.
No executable selector uses this component. Full linked-graph, affine,
ownership/private-profile verification, pure C emission and final-source paired
release acceptance remain required under #1142/#976.

## My shared ownership declarations

`src_nano/nvm2c/ownership.nano` implements the shared
`nvm_ownership_contracts_validate` boundary for versions 1, 2 and 3. I retain
exact status and `requires_verifier` results. The bounded model supplies retained
layout bytes, string identities, narrow function/parameter/result descriptors
and code for constructor-site checks. My pure cursor/path/extension transport
is in `ownership_wire.nano`; it preserves truncation, padding and framing order.

I check complete/resource layout authority, reference descriptors, path tables,
concrete union variants, dense global declarations, effect signatures and
backward-linked element types. Revision-3 constructor bindings must name exact
empty ARRAY instruction sites and matching element tags. I preserve the shared
validator's private ARRAY_FIELDS/MAP_FIELDS refusals; their separate private
projections and executable profiles still need implementation. Declarations
grant no borrow lifetime, storage initialization, handler or execution authority.

The ownership fixture exposed an array-slice prerequisite under #979. My
NanoLang checker now preserves the slice receiver's complete type and checks
the receiver and integer bounds; complete array mutation checks remain strict.
My C checker also rejects invalid operands before bytecode publication and
retains receiver TypeInfo at nested storage boundaries. The original slicing
test defaults still select the actual installed stages. Explicit compiler
selection lets development qualification name its real tools without aliasing
them as Stage1/Stage2 or deleting any product assertion.

My [paired declaration and slice evidence](evidence/nanolang-ownership-20261120/README.md)
matches 2350 shared C status/authority decisions across all five Linux/Darwin
routes, with 1415 explicit expectations and unchanged declaration budgets.
Both fresh C-seed-built NanoLang compilers pass the unchanged four-method
slicing suite and semantic refusal controls. These development images do not
establish installed-stage or raw fixed-point qualification.

## My private retained-layout structure

`src_nano/nvm2c/private_layouts.nano` supplies the ARRAY, mixed and MAP
retained-layout structure profiles. I check exact transport before semantic
fields, retaining the 256-layout, 65536-field and 16 MiB limits. My shared
`payloads.layouts_profile` parser keeps public decoding on its original grammar.

For forward-containing tables, ARRAY permits record DAGs with scalar/string/
array leaves. Mixed additionally permits scalar union nodes beside record DAGs;
record edges still connect records, and array leaves remain record fields. MAP
adds map leaves in records. Prior-only tables retain the C codec's distinct,
broader transport behavior. Later declaration and executable consumers own
their remaining checks.

These are immutable byte views: callers keep the input bytes immutable. They
do not emulate C allocation ownership or establish element types, ordinary/
resource classification, reference eligibility or execution authority. Private
declaration projections and whole-module profile admission remain required.

My [paired private-layout evidence](evidence/nanolang-private-layouts-20261120/README.md)
matches all 1748 cases on four profiles across five routes on Linux/Darwin,
including 4640 explicit status expectations, exact field/byte limits and
complete C-provider ASan/UBSan. The original public payload and dependent
module/ownership gates pass; I retain the two corrected oracle expectations.

## My private declaration projections

`src_nano/nvm2c/declarations.nano` projects ordinary ARRAY, mixed record/union
and MAP declarations in NanoLang. I retain their distinct described, unknown,
absent, invalid and limit outcomes. My shared ownership helpers select explicit
private layout/descriptor rules; the public validator retains its original
profile. A later malformed declaration still takes precedence over an earlier
unknown fact where the C reader continues checking.

I preserve conservative descriptor/union work precharges, 4096 element types,
65536 bindings, depth64 coloring, exact array-field coverage, union variant
partitions, canonical map scalar/pair usage, and both declaration work limits.
`variant` supplies checked union slices; `map_field` searches the complete
sorted binding table. I expose no partial inventories on refusal or absence.

These are immutable views and numeric facts, without C pointer/allocator
ownership emulation. I require immutable input. On my qualified64-bit hosts, the fixed cardinality ceilings keep the C
ordinary/mixed copies below16MiB and map sidecars below2MiB, so their default
copy-byte ceilings cannot add a refusal.
My NanoLang heap representation has its own costs. I retain the C refusal
decisions and logical work bounds without claiming identical heap usage or
allocation-denial behavior.

Private declaration success supplies no instruction-flow, initialization,
borrowing, handler or execution authority. Whole-module profile conversion and
executable verification remain required before representation analysis and C
emission may replace my reference translator.

My [paired declaration evidence](evidence/nanolang-declarations-20261121/README.md)
matches 3988 complete reports per profile on three profiles and five routes
on Linux/Darwin, including 1728 explicit statuses and fourteen original producer
models. Full C-provider instrumentation and original declaration/public
ownership regressions pass. I retain the initial oracle/expectation failures;
whole-module and product cutover gates remain open.

## My private profile routing

`src_nano/nvm2c/routes.nano` classifies public MAP routes, ordinary ARRAY
candidates, resource ARRAY routes, mixed candidates and service/File priority.
I keep the independent bounded layout and ownership scans: a malformed span
cannot hide a MAP identity already established by the other span. Kind5 is
recognized before its remaining header, revision, payload or padding can fail.

Valid absence and DEFER remain distinct. DEFER requires the original full
validators; it grants no fallback execution. Selected or invalid private routes
cannot pass as shared ownership. ARRAY/mixed candidates retain their early
returns, while owner ARRAY routing validates its complete layout scan and
rejects combined version3 authority. Service imports, bindings and actual File
instruction boundaries suppress those ARRAY/owner selectors; MAP classification
retains its separate priority.

I share File scanning through `instructions.has_file` and retain the module
helper as a wrapper. Routing does not validate declarations, code flow or
runtime authority. I integrate shared ownership conversion below; complete
private admission remains a separate dependency.

My [paired routing evidence](evidence/nanolang-routing-20261121/README.md)
matches5727 complete reports across five Linux/Darwin routes, with1315 explicit
expectations and fourteen original producer products. The declaration, core
module and instruction regressions pass; no routing result grants execution.

## My shared ownership module conversion

`module.decode_core` now carries section13 through whole-module decoding and
requires its ownership feature bit exactly when nonempty bytes are present.
I run the public MAP quarantine scan before table decoding. Selected private
profiles remain pending; malformed selected MAP metadata is an error, never a
fallback to shared validation.

Core table decoding, references and features precede narrow-carrier conversion.
I preserve byte-exact constant identity, parameter and first-result tags,
function/debug/code bounds, canonical retained-layout reconstruction and
per-kind counts. I then select ownership profiles and run shared declarations
before passive claims. Tolerated layout table tails are removed by canonical
re-encoding; an empty layout table is retained only when requested by the
retained-layout feature. Invalid owner routing returns the conversion error
before the shared validator's more specific statuses.

My [paired whole-module evidence](evidence/nanolang-ownership-modules-20261121/README.md)
matches completed decisions and exact ownership/passive bytes across five
Linux/Darwin routes. I retain2531 synthetic inputs, twenty explicit status
controls and fourteen original producer modules. Eight private inputs remain
pending on every route: six synthetic inputs and two actual nested-map
products. The269-case core module and2240-case passive-module regressions pass,
including their original products, prior-output refusals, widths and File scans.

This shared conversion path mirrors the C decoder's declaration boundary; it
does not run ordinary instruction verification. An explicit unknown-opcode case
therefore deserializes successfully on both sides without gaining execution
authority. Shared service14 decoding now follows the boundary below. Private
flow admission, ordinary executable verification and my NanoLang C emitter
remain required before cutover.

## My service wire codecs

`service_wire.nano` implements all five raw codec profiles: legacy File v1,
nominal File/TCP/WebSocket v2 and multi-instance v3. I decode complete tables,
check their normal narrow carriers and emit canonical bytes without calling
the C codecs. The caller selects the exact codec profile; this API does not
authorize section14 or choose a fallback after a malformed identity.

I preserve size/version/catalog/count/reserved/ordinal/index status precedence.
For v3 I check every entry's framing before semantic validation, then validate
catalogs, inactive slots and distinct active indices in instance order. Import
indices and layout indices have separate uniqueness domains; duplicates across
instances in either domain are errors. All64 instances and the4112-byte bound
remain intact. Failure publishes no partial decoded values or encoded bytes.
My arrays represent normal uint32 carriers, not arbitrary C pointers or overlap.

My [paired service-wire evidence](evidence/nanolang-service-wire-20261121/README.md)
matches9208 complete reports with7311 explicit expectations across five profiles
and five Linux/Darwin routes. I instrument all five actual C codec providers;
original C raw/nominal and allocation-denial controls retain their separate
scope. Fourteen real products from both producers preserve VM and instrumented
native results, including byte mutation and the unchanged generic/u8/conversion
fixtures. Raw success grants no module, code-flow, execution or host authority.

The fixture exposes prerequisites that I repair rather than change its byte
model. Native inference waits for caller storage before interpreting a computed
integer write (#1150); the shared C checker refuses known noninteger byte
writes, and its interpreter preserves byte identity and modulo256 mutation
(#1151). `bytes_from_string` now returns its advertised byte storage in NanoVM
(#1110). I retain all original failures and strict prior-output controls.

The genuine development NanoLang compiler remains outside `bin`. Its Linux GNU
assembler fallback therefore needs the real `NANO_AS_CAPTURE_HELPER` explicitly
configured (#1152). I retain that missing-companion failure, configure the
actual helper and pass the complete dependency shadows without changing capture
guards. This establishes a declared development runtime closure, not installed
stages or a raw bootstrap fixed point. Service metadata and complete profile
admission remain required before C emission and product cutover.

## My nominal service metadata plans

`service_nominal.nano` describes exact File, TCP, WebSocket and multi-instance
metadata after the caller selects raw profiles2 through5. I validate complete
catalog imports, exact nominal names and members, prior-only layout edges,
per-kind counts and version1 ownership descriptors before returning any rows.
Repeated instances retain distinct layout identities even when their catalog
names match. Unknown layouts retain zero ownership flags and their actual kind;
my wire kinds distinguish tuple1 from enum3.

`service_catalog.nano` contains generated immutable NanoLang data. Its generator
reads the canonical C catalog views at development time. My running NanoLang
reader calls no C getter and requests no per-program C plan. I reconstruct the
complete canonical catalog view from the pure getters in my route tests, and
`make check-nvm2c-service-catalog` checks generation parity.

I expose layout, instance/type, source and import queries. My unified private
queries preserve `instance*9+type` and `instance*5+method` identities, including
missing slots, catalog counts and TCP endpoint classification. These are
metadata identities, not host handles or code-flow admission. Invalid plans
return no partial rows. Normal immutable value carriers model the C APIs'
valid storage; original C tests separately exercise pointer and allocation
failure contracts. C allocation denial is not a claim about NanoLang's heap.

My legacy File v1 narrow and rich-v2 metadata adapters are implemented below.
My shared module decoder now connects this metadata boundary to ordinary
module framing and narrow conversion as described below. Private WebSocket
codec selection remains separate. This library does not change installed
products or translator selection.

## My shared service metadata boundary

`service_module.nano` mirrors the shared normal narrow-module service validator.
I validate legacy File v1 import names and positional signatures using the pure
immutable catalog. I dispatch nominal versions2/3 before legacy fallback,
selecting exact File/TCP or multi-instance plans. A standalone WebSocket v2
claim remains refused by this ordinary shared boundary; its existing private
transport adapter is distinct. Multi-instance plans can describe WebSocket
instances under their own exact catalog rules.

I preserve the existing NvmV2Result status order: a File instruction without a
nominal prefix refuses with8 before absent metadata can succeed; malformed raw
legacy bytes refuse with8 before import/reference/callback counts refuse with14;
a legacy index outside the exact import table refuses with13 before that
entry's name/signature checks. Nominal plan limits map to13 and other metadata
refusals to8. Legacy validation does not inspect unrelated layouts/ownership.
Nominal metadata cannot silently fall back to a legacy plan.

I expose metadata presence and bounded File-instruction classification. I scan
only in-range function spans, recognize File opcodes before their operands and
stop at unknown or truncated non-File instructions. An immediate byte equal to
a File opcode is not an instruction claim. Pending execution is the union of
metadata presence and File instruction presence; successful metadata validation
does not remove it.

My pure input arrays represent canonical normal storage: absent service bytes
have no separate nonnull empty pointer. C-only malformed pointer/count graphs,
allocation failures and destination preservation retain their original C test
scope. Shared whole-module service conversion follows the boundary below;
full flow/host/execution admission remains required before product cutover.

## My rich service metadata adapter

`service_v2.nano` accepts typed constant, signature, function and import tables,
canonical normal layout rows, code and extension bytes. I preserve the shared
rich validator's legacy signature-index ordering, nominal preflight limits,
prior-layout-edge checks, STRING constant requirement and referenced signature
storage checks. I reuse my pure narrow validator after conversion. I scan the
original function spans for File instructions before dropping code offsets
from the metadata-only narrow view.

I expose ordinary shared validation and the separate private WebSocket adapter.
Private WebSocket success does not change ordinary shared transport selection.
My presence query includes the service feature bit, raw bytes and service-kind
imports. Classification and metadata validation grant no executable authority.
My arrays represent normal C storage; they do not model arbitrary invalid
pointer/count graphs. I explicitly exercise absent signature storage. Existing
C tests retain allocation-failure and malformed-pointer coverage.

The component fixture compares actual C rich-provider status and classification
against five execution routes, including actual File/TCP/WebSocket/mixed source
products. Paired qualification and its retained failure evidence belong to the
roadmap receipt; this implementation is not an installed compiler cutover.

The same fixture exposed constructor and value-transport defects. I register
internal imported records with their canonical owner from first registration,
check declared record returns before publication, and carry typed/scoped
conditional and ordinary record-match values through both producers. My native
record join widens exact scalar fields into tagged destinations while preserving
missing tags and payloads. Exact incompatible and aggregate fields remain
refused. These repairs do not authorize resource record results or numeric
array-write joins under #1148.

## My cutover gate

I must execute my translator both in NanoVM and as a standalone native product,
then compare the complete pinned required source/NanoISA corpus across both
frontends and both execution routes. That includes malformed modules,
prior-output preservation, nominal/collection/callable identity, evaluation
order, traps and cleanup. I must translate the actual compiler and translator,
retain raw Stage1/Stage2 byte equality, and pass installed/no-C-seed execution
on Linux and Darwin. I inspect the resulting process and link closures for
legacy translator or embedded VM dependence. The existing C translator may be
used as a test oracle during development; it cannot perform production
translation after cutover.

I keep every unfinished payload, verifier, analysis, emitter, service, driver,
bootstrap and platform requirement open until its own acceptance passes.


## My shared service module boundary

`module.nano` retains raw service14 bytes in `ModuleView.services`. I preserve
section-directory framing order: legacy56-byte, nominal File120-byte,
nominal TCP128-byte and bounded version3 multi-instance service frames. The
public decoder refuses a standalone WebSocket104-byte frame; its private codec
requires its own future entry point and qualification.

After structural table decoding I adapt typed constants, signatures, function
spans, imports and canonical re-encoded layouts to `service_v2.nano`. Rich
service preflight and exact service feature consistency run before ordinary
cross-table references. Function names/ranges, callbacks and the remaining
features keep their original order. Successful shared service presence also
triggers the existing narrow conversion checks, including duplicate/string
constants, narrow spans/results/debug fields and canonical retained layouts.

I run narrow service validation before ownership/passive declarations. Valid
nominal File/TCP/multi metadata preserves descriptive ownership bytes without
passing it through the ordinary shared ownership validator; legacy File keeps
that validator. This matches the C transport boundary and grants no instruction
lifetime or host authority. I continue to quantify unimplemented private
ARRAY/MAP decisions instead of accepting them through this path.

My current Linux development gate matches704 synthetic complete decisions and
exact service/ownership/passive bytes through five routes, including128 explicit
C expectations and directory/error-priority controls. A separate retained eight
actual C/NanoLang File/TCP/WebSocket/mixed producer-product gate passes the same
five routes. The combined permanent fixture includes both groups. My later full Linux selection passes all five methods in1410.818s, including712 shared-service decisions. The [bdffa4ccc Darwin checkpoint](evidence/translator-components-bdffa4ccc-darwin-20261121/README.md) passes the original five-method module/claim gate in1090.333s, but the service test deletes one TemporaryDirectory containing its producer products. I retain that result and mark the missing products explicitly; the historical deleted products remain unavailable. My later exact-ad2
[paired retained selection](evidence/conversion-projection-darwin-20261121/README.md)
covers all five original methods and preserves every actual producer product.
Linux passes all five in1393.071s. Darwin retains three passes from its first
setup-failed observer and passes only the two missing methods in698.042s after
supplying prerequisites, preserving the first failed manifest. Both hosts match
all712 shared-service decisions through five routes. This new pair completes my
descriptive shared-service retention requirement. Full executable/private-profile
verification, pure C emission and final-source release gates remain open.
I retain the initial rich-only UINT32_MAX span fixture refusal, isolated
name-mutation correction, default-shadow deadline refusal, successful traced
169-shadow observation and excluded stale-artifact diagnostic. A later success
does not identify the historical timeout's cause or establish reliable source
creation. Product selection, the pure C emitter and installed stages remain open.

## My function decode boundary

`functions.nano` decodes every primary instruction in a bounded function span,
including unreachable bytes. I retain relative starts/next offsets, exact
primary operands, terminal and instruction boundaries, one-based instruction
indices and resolved targets. Branch targets are relative to the opcode byte;
resolved branch addresses include the function's code offset. Direct and tail
calls resolve to function indices. I refuse malformed or truncated instructions,
interior/out-of-range branches and calls outside the declared function count.

My terminal boundary is valid for ordinary branch decoding, but it names no
instruction. A File-result branch requires both its target and fallthrough to
name bytes before that terminal boundary. Handler decoding preserves its
ordinary terminal-boundary behavior; executable handler-arm validation belongs
to the verifier. I clear every partial view on refusal. This immutable decode
boundary grants no stack, type, lifetime, effect or host authority. My Linux test matches1325 complete reports through five routes, with1197 explicit validity expectations, using the actual native Stage1 source compiler. My retained [bdffa4ccc Darwin checkpoint](evidence/translator-components-bdffa4ccc-darwin-20261121/README.md) passes the complete original gate in13.530s. I retain its available generated products and complete reports; exact source providers remain separately identified. Full verifier/profile integration and final-source release qualification remain required.

## My copied record instruction recipes

`record_instructions.nano` describes the existing ARRAY/MAP instruction copier
before full profile analysis. I retain all94 ordinary record/array opcode recipes
and the six additional MAP operations. My recipe kind and obligation bits copy
the existing root, tag/bounds/nominal, safepoint, operand-staging, retain-before-
release and first-error cleanup requirements. These are runtime obligations;
I have not discharged them by copying their names.

I use my pure function decoder, then resolve literal/aggregate field counts,
call arities/results and current-function returns into exact stack counts.
CALL retains its continuation and a separate callee; TAIL_CALL and RET have no
successor. Conditional branches retain target before fallthrough. I copy every
instruction, including dead code, with relative edges and lossless operand
halves. My descriptive bounds are256 functions,65536 instructions,256 stack
operands/results and256 derived global slots. I clear partial output on refusal.

This low-level copier preserves its existing unsigned global-index increment,
including UINT32 wrap. Full profile analysis separately rejects invalid globals
before execution. A copied result does not establish valid globals, types,
origins, lifetimes, stack-flow consistency, complete resource accounting or
executable admission. Inferred MAP key/value/result tag sets also remain part of full profile analysis.
I keep those gates open; this module is not selected by
my product driver or whole-module decoder.

My Linux differential gate matches2014 reports through direct C-seed native
execution and both producers in NanoVM/sanitized C AOT, with994 explicit validity
expectations. Reports contain the relative instruction descriptors and operand halves for functions
up to1024 instructions; the65536/65537 boundary controls return the final
validity/count summary after complete copying. Both producers execute useful
shadows under their existing component deadline. My original identifier gate
passes five controls and3308 static fragments. My separate [bdffa4ccc Darwin checkpoint](evidence/translator-components-bdffa4ccc-darwin-20261121/README.md) also passes the complete instruction-recipe gate. The exact source providers remain distinct; full checked-profile analysis and final-source qualification remain required.

## My closed recipe stack-height walk

`record_stack.nano` checks reachable stack heights over freshly decoded
ARRAY/MAP recipe inputs. I start with an empty operand stack, reject underflow
and unequal-height joins/backedges, and require exact declared result counts
at explicit and implicit exits. I measure both sides of each reachable
instruction and reject a locals-plus-operand frame above65535 slots. I include
callee results in a tail exit's maximum, matching the shared height helper;
my copied execution recipe itself records no pushed tail result.

I preserve dead-code behavior: decoding and recipe framing still inspect it,
but an unreachable valid POP does not cause reachable underflow. The shared C
height helper accepts an empty body independently of its declared result;
structural verification owns that separate refusal. This entry also does not
establish matching tail-return signatures, value tags, origins, lifetimes or
complete resource budgets. Handler/effect/explicit-retain flows require the
broader verifier. I grant no module or executable admission from a height result.

My [Linux height checkpoint](evidence/record-stack-linux-20261121) matches1475
reports per route,963 explicit expectations and512 randomized CFGs through five
routes in181.116 seconds. Targeted actual C copier/shared-verifier sanitizer
controls and original identifiers pass. I retain the initial DELETE/LEN fixture
opcode error and correct only its byte. My separate [bdffa4ccc Darwin checkpoint](evidence/translator-components-bdffa4ccc-darwin-20261121/README.md) also passes the complete height gate. The exact source providers remain distinct; full profile/verifier integration remains required.

## My ordinary height and explicit-retain walk

`stack_heights.nano` reuses `functions.nano` and matches the ordinary shared
`verify_stack_heights` helper. My entry is
`verify(code, start, length, signatures, imports, current, locals)`. A signature
contains unsigned16-bit arity and unsigned8-bit result count; an import contains
unsigned16-bit parameter count and unsigned8-bit return tag. I distinguish safe
decode/precondition refusal (`Result.decoded = false`) from a completed helper
call. Successful results report the reachable maximum; failures carry the exact
shared diagnostic and publish no maximum (`maximum = 0`).

I preserve dynamic direct/indirect/module/foreign calls, capture and aggregate
literal counts, PERFORM, PICK and ROLL. I walk branch targets before fallthrough,
visit each instruction once and require equal stack height and explicit retain
balance at joins. GC_RELEASE must first have its operand, then a held reference.
I preserve explicit and implicit return shape checks, handler installation with
balanced ownership, empty handler-target entry, EFFECT_RESUME constraints and
16-bit locals-plus-operand frame bounds. A handler targeting the terminal byte
retains the original outgoing-height end check. HALT and TAIL_CALL retain the
helper's existing termination behavior. FILE_RESULT_BRANCH has only the ordinary
fallthrough height edge here; its execution profiles own additional checks.

I decode dead instructions, but their unreachable stack/retain effects do not
participate in the walk. An empty body passes this helper independently of its
declared results; full structural verification owns that refusal. My preconditions
validate representable signature fields, current-function/local bounds and every
CALL_EXTERN index before the helper can dereference it. These prerequisite
refusals are normalized to `D` in the differential carrier. They do not replace
all module table, signature, effect, ownership or profile validation. The carrier
limits function/import tables to65536 entries for bounded tests; my pure entry
retains the unsigned32-bit count boundary.

My [Linux checkpoint](evidence/stack-heights-linux-20261121/README.md) matches3154
complete reports per route,546 explicit expectations, every primary byte,
1024 randomized CFGs, nonzero spans, terminal/interior/dead branches and16-bit
signature/depth/frame boundaries. I run five routes using the actual guarded
bdffa4ccc native Stage1 as NanoLang producer; the full gate passes in68.869s.
Both generated native routes retain ASan/UBSan. Targeted instrumentation of the
actual C height helper, decoder and ISA reader matches the same3154 reports;
all97 original C verifier tests and the original identifier controls pass.
I retain my initial new shadow's mistaken GC_RELEASE error priority and correct
its intended operand without changing either verifier algorithm.

I compare successful-allocation, representable helper behavior. Allocation
failure/publication policy, counted private resource limits, full ordinary
verification and ARRAY/MAP ownership/origin/lifetime analysis remain separate.
My [same-source Darwin checkpoint](evidence/ordinary-verifier-components-darwin-20261121/README.md) passes all3154 reports through five routes in26.199s, with identical Linux corpus bytes/reports and3558 guarded inputs. Targeted actual C helper sanitizers pass there; I claim no Darwin LeakSanitizer coverage, product selection or pure C-emitter completion from this component.

## My shared shallow operand-type walk

`stack_types.nano` reuses my ordinary `functions.nano` decoder and implements the
shared C verifier's shallow tag analysis. I retain the primary instruction
rules, exact mismatch diagnostics, branch-target-before-fallthrough FIFO order,
queued-state deduplication and widening of unequal tags to unknown. My ISA
schema generates the diagnostic opcode names; no runtime C verifier call is
part of this NanoLang implementation.

My entry is `verify(code, start, length, function_count, current, maximum)`.
`Result.decoded` distinguishes an invalid input/function frame from a completed
helper call. A decoded result with `valid = false` carries the original typed
mismatch message, including function index, opcode, relative byte offset,
expected tag and actual tag. My differential carrier normalizes decoder
refusals to `D`; I compare typed mismatch messages exactly.

I preserve this helper's limits of knowledge. A zero maximum, a maximum above256
or an empty body skips type checking after decoding. Unknown values never cause
mismatches. Locals and dynamic call effects do not supply declared type facts;
dynamic stack effects clear the tracked state. DUP and SWAP do not acquire new
rules in this pass. I accept u8 identity in CAST_U8 and retain both integer
results of carry/borrow/wide multiplication. Generic ADD0x20 remains distinct
from typed I64_ADD0xc0. My handler edges retain the shared helper's empty-entry
behavior; FILE_RESULT_BRANCH keeps only the helper's ordinary fallthrough type
edge after its decoder checks.

I require the independently computed height and all structural/metadata checks
when composing the full verifier. Successful shallow analysis alone grants no
height, signature, ownership, origin, lifetime, host or execution admission.
The legacy unbudgeted C entry may skip analysis on allocation failure; this
component checkpoint compares successful-allocation decisions. Counted private
ARRAY/MAP resource accounting and allocator-failure/publication behavior remain
separate work.

My Linux corpus covers10287 reports per route,323 explicit expectations, all256
primary rule/name slots,512 randomized CFGs and directed decoder/handler/deep-
stack cases. An observation through the unchanged C helper's existing work
counter confirms four queue-wrap cases:8/11,23/41,71/137 and264/523 instruction/
visit counts. The counted side entry preserves every ordinary decision and
message, returns its reserved bytes and reaches no budget refusal in this
corpus. These observations establish test coverage, not private budget admission.
I retain targeted ASan/UBSan/leak checks of the actual C type helper, function
decoder and ISA reader separately from full-provider qualification. My exact
products, source-provider identities and terminal outcomes belong in the
[Linux checkpoint](evidence/stack-types-linux-20261121/README.md). My [same-source Darwin checkpoint](evidence/ordinary-verifier-components-darwin-20261121/README.md) passes all10287 reports through five routes in42.217s, with identical Linux corpus bytes/reports and3558 source/test/spec/Make hashes. Targeted actual C helper sanitizers also pass on Darwin; I claim no Darwin LeakSanitizer coverage. Full verifier/profile integration and pure C-emitter cutover remain open under #1142.

## My composed ordinary function-body checks

`function_body.nano` composes my ordinary function decoder, operand checks,
height/explicit-retain walk and shallow type pass in the same order as the shared
C `verify_function_body`. I validate operands even in dead instructions before
walking reachable stack/type states. I preserve tail-call result count/tag,
linked module presence and declared arity/result counts, capture counts, string,
import, global, local, upvalue, aggregate-layout and tag bounds. Handler names,
parameter slots and executable target arms retain their original priorities.
Direct/branch decode refusals still precede operand checks; complete operand
checks precede height failures, which precede type contradictions.

My entry is `verify(code, start, length, context, current)`. A context supplies
function arity/result/tag/local/upvalue facts, import parameter/return facts,
optional linked-module function signatures and string/record/enum/union counts.
I first require representable fields and safe table/index/span inputs. These
preconditions do not assert the full module's signature consistency, declared
callee bodies, import ABI, retained layouts or execution policy. A missing
linked table preserves the shared helper's structural-only module-call behavior.

`Result.decoded = false` reports a precondition/decoder refusal. After decoding,
`supported = false` and `valid = false` mark my explicit affine boundary at the
instruction where the C helper invokes lifetime analysis. Earlier ordinary
operand failures keep their priority. I do not implement or approve that
lifetime analysis here. Supported ordinary failures retain exact C diagnostics;
success reports the proven reachable maximum. Full module structure, profile
selection and resource/lifetime admission remain required before execution.

My [Linux checkpoint](evidence/function-body-linux-20261121/README.md) matches6935
reports through five routes using the genuine guarded bdffa4ccc native Stage1
producer in131.621s. I retain1184 independent expected reports, all256 primary
bytes,2048 randomized bodies/height CFGs, linked/unresolved/signature boundaries,
nonzero offsets, malformed/dead inputs and explicit diagnostic-priority cases.
I distinguish1779 passes,4279 ordinary failures,691 decoder/precondition refusals
and186 affine deferrals. My carrier normalizes only precondition/decode refusals
to `D` and the deferred affine branch to `A`; it retains the actual C affine
messages in stderr, and compares ordinary failures and maxima exactly.

Targeted actual C body/height/type/decoder/ISA ASan/UBSan/leak checks match all6935
reports; both generated native routes stay sanitized. All97 original verifier
tests and the original identifier controls pass. I retain my initial new-source
parenthesis refusal and the linked-shape coverage gap: a module-call stack
prefill reached linked validation before the intended call. I correct that
fixture with an indirect-call effect and add explicit shape/tag/handler
expectations without changing the C verifier.

I compare successful-allocation helper behavior with no private budget selected.
This is not full affine/owned/profile verification, pure C emission or product
cutover. My [same-source Darwin checkpoint](evidence/ordinary-verifier-components-darwin-20261121/README.md) passes all6935 reports through five routes in50.974s, with identical Linux corpus bytes/reports and3558 guarded source/test/spec/Make inputs. Targeted actual C helper sanitizers pass there, with no Darwin LeakSanitizer claim. I preserve the first observer-only missing Make-target setup refusal and all frozen roots/primary dirty work/stashes.

## My empty-result structural boundary

Under #1174 I repair a gap in ordinary module verification. Atbf5b438c442c,
an empty main declaring one int result passes actual verify-only,
then the actual VM refuses its implicit return with zero values instead of one.
My earlier height-component statement that structural checks owned this refusal
was an incomplete description of that source: the helper accepted the empty
body, and the ordinary structural guard was missing. Private record/array
structure already rejected this case.

I now reject an empty result-bearing function in the shared module-range check,
after existing range/name/tag/parameter/signature/local checks. I preserve the
empty-height helper, valid empty void functions and zero declared-depth absence;
a zero declaration is still not full verification. Non-entry functions receive
the same structural check. The C translator already refuses the invalid required
entry and preserves prior output. Public tools already refuse legacy NVM01 wire
files; I keep that separate refusal rather than enabling an obsolete format.

My [Linux checkpoint](evidence/empty-function-results-linux-20261121/README.md)
passes101 verifier tests, the complete original569591-assertion VM target,
four actual product tests and targeted actual C verifier/type/decoder/ISA
ASan/UBSan/leak controls. A real empty-void source body passes through both
producers/five routes using the guarded qualified bdffa4ccc native Stage1.
I preserve the baseline tools/bytes/runtime failure, red gates, first legacy
fixture expectation and sanitizer-command observer setup refusal. My [paired Darwin checkpoint](evidence/empty-function-results-darwin-20261121/README.md) at9fc32e45d passes the same101 verifier/569591 VM assertions, four product methods/five source routes and targeted actual C sanitizers, with no Darwin LeakSanitizer claim. All3559 guarded inputs and143 binary carriers match Linux. The retained older Darwin VM reproduces the identical pre-fix failure. A post-test observer omits its re import; I preserve that failure and complete only pairing/archival without rerunning tests. This closes the observed structural gap; full module/profile composition and final-source release qualification remain required.

### My ordinary linked-module composition

I expose `module_verifier.verify_linked(bytes, links)` and
`module_verifier.verify_graph(bytes, links)` for ordinary checked V2 modules.
Each `LinkedModule` contains `resolved` and `bytes`. I ignore the bytes of an
unresolved slot. A resolved slot must load and satisfy its positive declared
stack obligations even when the root never calls it. I load the root first,
then resolved slots in source order, before graph policy or full body checks.

`verify_linked` checks the root against the supplied flat table. Its successful
`complete` result describes that ordinary root query. It does not certify every
target body. `verify_graph` additionally checks each present target against the
same table, in slot order. Its `GraphResult.module_index` is zero for the root,
slot index plus one for a failed target, and minus one for success. A successful
graph result retains the root's depths; every refusal discards partial depths.
This wrapper does not discover imports, recursively load dependencies or grant
host authority.

I retain the public C order for ordinary service and ownership policy before
root structure/body checks. I compare encoded cross-module call shapes against
resolved target signatures; unused unresolved slots remain permitted. My
ownership scanner stops at an undecodable instruction, as the C policy scanner
does. The ordinary body decoder still rejects malformed target bodies when I
query those bodies. Private layout profiles continue to defer at checked
preparation; full owned/private admission remains required.

My linked differential gate uses actual V2 inputs and public C deserialization,
conversion, declaration and linked-verifier APIs. Its complete-graph oracle
calls that same linked API on every present target. The binary test carrier's
64-slot bound belongs to the harness, not to my production API. Two legacy
in-memory controls with a callee arity above65535 or result count above255
cannot survive V2 conversion and stay separate from the materialized corpus.
My [Linux checkpoint](evidence/linked-module-verifier-linux-20261121/README.md)
passes304 carriers in both modes through five routes and six fresh original
producer targets. Five carriers exercise harness refusal and four exercise
explicit private deferral;295 ordinary V2 decisions retain public C parity.
The complete7132-module/two-mode/five-route unlinked regression also passes
after the shared preparation/context change. No executable selector uses these
APIs. My [same-source Darwin checkpoint](evidence/linked-module-verifier-darwin-20261122/README.md) passes both full gates in658.533/1099.153s, with every corpus byte and all48 normalized report files identical to Linux. Targeted actual C sanitizers, original identifiers and all3567 guarded source inputs match; the three original examples are separately guarded. I preserve all setup refusals and original roots. Private/affine verification, pure C emission and final-source release qualification remain required.


### My affine reference-place dependency

I implement `reference_places.nano` before the remaining local affine state and
bytecode walk. My immutable place retains a nonzero invocation as two unsigned
32-bit words, a local identity, claimed root/referent, a bounded field path and
shared/exclusive mode. Explicit presence facts preserve the C descriptor's
absent-pointer refusals; an empty path or empty record needs no field storage.
The API expects native-width local/path/tag/layout words from checked facts.

My validity query checks the separately authoritative root layout and every
reachable scalar-only record sibling, not just the selected path. Layout edges
must point strictly backward; the selected path must end at a record containing
only numeric/bool scalars. A descending marked walk visits shared subgraphs
once. I do not accept strings, arrays, unions, forward edges or merely claimed
nominal authority through this reference contract.

My overlap and conflict queries inspect descriptors independently of layout
validity, as the C APIs do: invocation/local identify the owner, and a common
path prefix overlaps even when a claimed layout is false. Invalid descriptors
conservatively conflict. Shared holds coexist; exclusive holds block reads and
writes, and direct owner writes/moves conflict with either hold mode.

My actual C oracle calls the public reference-place APIs without duplicating
their policy. My test carrier bounds layout rows at65536 as a fixture budget and fields/path
entries at the native uint16 limit65535; its D refusal describes the harness, not a
new product acceptance decision. My [paired exact-source qualification](evidence/reference-places-darwin-20261122/README.md)
matches1629 ten-bit actual-C reports with487 independent expectations through
five Linux/Darwin routes in26.588/10.692s. Every corpus byte and all six
complete report files match, along with3571 guarded source Git blobs. All75
original C checks, actual implementation/oracle/unit ASan/UBSan and identifiers
pass on both hosts. I preserve all frozen providers and primary audit/stashes. I keep affine state/bytecode, private-profile composition and runtime
admission open, and no executable selector calls this new component.

### My shared affine local-state replacement

I implement `affine_facts.nano` and `affine_state.nano` after the paired place
checks. Facts preparation first runs the complete existing shared ownership
validator, retains ordinary decoded layouts, function/result/local descriptors,
layout flags, exact union variant slices, effect signatures and typed-array
rows. Private checked-state constructors in the mixed/owned-array analyses are
not admitted by this shared factory.

My owning analysis allocates a distinct positive module identity for each
module object and a distinct analysis identity for each preparation. Clones
and lexical handler arms retain the latter; independent preparations do not
become equal merely because their bytes match. These identifiers belong to
my verifier context, never to producer-supplied ownership bytes. Facts and
returned state records are internal immutable views; they are not authority
certificates that arbitrary callers may forge or mutate.

My transition API returns a new state on success and the original state on
refusal. It preserves exact owned obligations, initialized copyable values,
selected union arms, regions, parent references, invocation/caller origins and
handler resume facts. Joins intersect copyable initialization and forget a
selected union arm only after exact live resource/reference obligations match.
Queries return explicit validity and expose no partial values after refusal;
value-type arguments use mode-zero slots, while declaration slots retain
reference modes. No operation performs bytecode or runtime execution.

I retain the first690 native traces, whose full actual-C snapshots agree even
though five handwritten expected booleans were wrong. The expanded3058-trace
preflight passes all five C/NanoLang native/VM/sanitized routes in239.392s.
The unchanged C state target passes563 ordinary and634 allocation-failure
checks. These preflights are not a frozen final-source qualification.

A separate input-path mutation witness then catches an alias in my first
port: a borrowed place retains the caller's path array. The actual C witness
passes16 checks, while my original source fails two ownership assertions. I
copy paths at installation, cloning and caller binding, and extend the oracle
and driver with restored path-perturbation isolation checks. I retain the
failed witness and explicitly identified reverse-patch reconstruction of the
uncommitted pre-copy draft; I do not call that reconstruction a frozen source.
My [exact3df7ac29e paired qualification](evidence/affine-state-darwin-20261122/README.md) then passes3141 traces through five routes in224.344s Linux/79.364s Darwin, with383 independent expectations. All corpus bytes, six complete report files and3577 exact Git source inputs match. All563 ordinary/634 allocation-failure C checks, targeted actual C ASan/UBSan and identifiers pass. Independent readers verify12177 Linux/7213 Darwin archive records; every source/tool/frozen-provider/primary-audit/stash guard holds. Darwin ASan disables leak detection; I claim no Darwin LeakSanitizer coverage. No new raw bootstrap executes.

My oracle includes the actual C state implementation and invokes its public
transitions/queries. It snapshots retained state and checks failed operations
against an actual C clone, preserves query output sentinels, and restores its
private path perturbations. Its analysis/module identity adapter models C
Facts/module object identity rather than replacing it with content equality.
The test carrier bounds sixteen simultaneous state handles,65536 reference
slots/commands and native operand widths. Harness D is distinct from a state
transition refusal. Affine bytecode/CFG analysis, private profiles, pure C
emission and complete release acceptance remain open; no executable selector
uses these local transitions.


### My CFG frame and scheduling foundation

I implement internal frame joins in `affine_frame.nano`, a bounded FIFO in
`affine_worklist.nano` and initialization/lexical-return relation algebra in
`affine_global_flow.nano`. My frame stack retains exact owned/observation tags,
nominal/array types, observation roots, selected union arms and all 256 excluded
arms. Joins preserve handler key/site order, floor and activation, then intersect
copyable local initialization and union the global relations. I copy nested
mask/local/global storage before successful transitions; refusal retains the
input. These internal records are not wire-format authority certificates.

My Linux relation/FIFO gate passes all five C/NanoLang native/VM/sanitized routes
against actual C: 168386 operations, every pair of 289 normalized flow states,
and seven full/wrapping/deduplicating queue traces with capacities through4096.
My frame corpus adds665 cases. All1001 combined carriers match actual C through
C-native and both source-producer VM routes, including immutable-input and
nested-clone checks. I retain my first driver cond-brace parse refusal and its
ordinary-if correction.

My initial complete frame five-route gate failed under #1202: C AOT refused
an optional-to-exact-Boolean aggregate conversion. Both actual producer products
retain that refusal, and a seven-line frame-library import reproduces it.
Read-only GDB identifies a deferred Boolean array read whose result is equated
with a scalar local. I preserve the exact-destination refusal tests and every
original assertion and deadline. I retain that initial failure as provenance
for the later repair.

My localization reduces the barrier to a twelve-line source with a mutable
Boolean read, array write and short-circuit and/or copy. Sixteen basic neighbors
and all unary-not variants translate. DUP/PICK reuse the local's shape and then
seed an exact scalar type; LOAD_LOCAL already retains inferred scalar storage.
My original 64-source/PICK regression retains nine valid translation failures;
strict invalid-depth and prior-output controls pass. The unchanged graph API
passes 312 counterfactual, exact-target and mismatched-payload checks normally
and under ASan/UBSan. My [localization receipt](evidence/scalar-copy-localization-linux-20261122/README.md)
retains the then-unapplied one-line proposal.

Under the [narrow #1202 handoff](https://github.com/jordanhubbard/nanolang/issues/1158#issuecomment-7093099516),
I extend only the existing inferred-scalar `LOAD_LOCAL` condition in
`sim_push_slot` to `DUP`/`PICK` at source
`0962d3c9dbf4376b2fac82b8f04f0c24e0ccd8b3`. All 64 source
copies and valid PICK0 translate. Invalid PICK1 retains assembler/verifier/C
refusal and prior output; all 312 graph and 3890 original shape checks pass
normally and under ASan/UBSan. I relax no merge function or shape API.

My [Linux copy-admission receipt](evidence/scalar-copy-admission-linux-20261122/README.md)
at `1b83cdabd3c8cd3103f24b5e6b0a5b639c41a68c` completes the original frame gate:
all 1001 carriers match actual C through C-seed native and both source producers'
VM and sanitized-native routes. Every complete report has SHA256
`f5f4555c26ea20dbc13237f4a9843c42e98b16f7fc6f4e9f9ea3de7b3d3dcbd0`.
Both original methods pass with zero failures, errors or skips in 422.750 seconds,
retaining all 665 frame cases, 168386 relation operations, 289 normalized states
and seven FIFO traces. The independent reader verifies 4711 records and 1384
exact Git inputs, the one-line change, original/candidate consumers and strict
negative controls. Generated C is instrumented and links the retained runtime
object; I do not claim whole-runtime instrumentation or fresh compiler products.

This closes the scoped Linux #1202 copy repair and qualifies internal primitives
at the recorded source/tool pins. Later integrated revisions require separate
qualification. Current-source Darwin, whole-function instruction transfer and
convergence, handler/global/effect and private-profile composition, pure C
emission and executable selection remain open, alongside the separate #1204
ownership-query/return-merge barrier and full release.


## My C-text component

`src_nano/nvm2c/c_text.nano` spells tokens without admitting a module or
calling a C translator/analysis API. `bytes_literal` returns `{valid, text,
length}`. Its fixed-width three-digit octal escapes preserve every byte and
cannot consume a following digit. Empty input has a valid quoted empty literal
and length0; counts above16777216 return invalid with empty text/length0. A C
string literal includes a trailing terminator; consumers retain the explicit
input length instead of using `strlen`.

`unsigned32` rejects values outside0..4294967295. `unsigned64(low, high)` checks
both halves before spelling unsigned64 shifts/OR, without forming an overflowing
signed NanoLang value. `signed64` treats the minimum endpoint before negation.
`name(kind, ordinal)` accepts kinds0..5 for function/local/label/string/global/
module and ordinals0..4294967295, using separate generated prefixes. Rejected
integer/name queries return empty text, distinct from every valid spelling.

My [Linux component receipt](evidence/darwin-current-source-20261124/diagnostics/c-text-boundary-linux-20261124/README.md)
at sourcee5b20896b343 retains18702 independently verified records. Both methods
pass through five existing routes in13.422s. I compile and execute every route's
emitted C11 tokens with strict warnings and UBSan; the complete81554-byte result
matches SHA256 e3eff8151eddff731c6825c70a866d33e91e86ef9e904c61fa7eda18f62baabe.
All256 octets, embedded zero, UTF8/non-UTF8 bytes, block boundaries,65536-byte
payloads, integer endpoints/range refusals and generated identifiers participate.
All five routes refuse an actual oversized16777217-byte input. The exact maximum
count is tested by a shadow, not a full maximum-size literal construction.
Source-consumer generated C is instrumented; the retained runtime object is not
whole-runtime instrumentation. The chosen NanoLang source producer is the actual
qualified16b Stage2, never a new current-source compiler claim.

I retain the original legal `signed` local's invalid C-native setup and the
subsequent report parser's bitwise-OR delimiter failure. #1254 has actual
local/parameter/nested-shadow generated-C refusals, successful NanoISA execution,
a safe-name native control and preserved prior outputs. Renaming only the
fixture to `signed_values` and fixing fixed-field parsing does not repair that
backend defect. Its source and original expectations remain retained.

This component grants no affine/profile/representation authority and emits no
complete translator. Full C control flow/runtime/service emission, complete
corpus/platform qualification, fresh products and actual cutover remain required.

## My global-transfer implementation contract

I next implement LOAD_GLOBAL and STORE_GLOBAL as an internal pure transfer,
using checked `ownership_queries.Global` declarations and affine local-state
type ownership. This consumes an already established value-call-graph decision;
it does not establish that graph or validate a global declaration table.

My result carries a fresh frame and a fresh vector of allowed function-entry
states. I preserve all local, stack, handler, activation, exclusion and untouched
global facts. Refusal returns isolated input snapshots and no successor. Before
transferring, I require matching declaration, frame-relation and allowed-mask
vectors of at most256 entries, four-bit relations and two-bit allowed masks.
Unsupported opcodes remain explicitly unhandled.

I preserve my C step's order: require the value graph and declared slot; refuse
resource-bearing types; then either load the exact declared value or require a
store value with exact tag/layout and no observation/owned escape. Loads
intersect the function's allowed-entry mask with the preimage of initialized
states. Immutable stores intersect it with the preimage of uninitialized
states; mutable stores keep that mask. Stores compose the slot relation with10
and pop exactly one value. With effect-global analysis selected, I leave these
allowed masks unchanged: the separate activation/handler proof owns those
preconditions. An empty allowed mask is a computed precondition, not a step
error. Full stacks retain the original load failure.

I compare the unchanged actual C `step` through a separate fixture, preserving
its production source and every old protocol and assertion. The fixture
normalizes failed transitions to isolated input snapshots, as my scalar/local
fixtures do, and reports both complete frame facts and allowed-entry masks.
Independent expectations cover every relation/allowed/mutability/effect mode,
physical tag and exact owner/nominal/array boundaries, stack limits, missing
slots, caller aliases and malformed carriers. All source routes, generated
sanitisers, targeted actual-C sanitizers and paired qualification remain
required. I retain any ownership-query dependency refusal under #1204 rather
than replacing that shared declaration type with a narrower substitute.

This is an implementation contract, not executable qualification. Global
load/store transfer does not complete interprocedural summaries, effect
activation/return semantics, CFG convergence, private admission, C body/runtime
emission, product selection or the full69-entry release.

## My complete affine array-transfer contract

I port all nine cases in `affine_array_flow.inc` as one internal pure transfer:
ARR_NEW, ARR_LITERAL, ARR_PUSH, ARR_POP, ARR_GET, ARR_SET, ARR_LEN, ARR_SLICE
and ARR_REMOVE. I consume decoded operands, checked local type facts, raw
ownership bytes and their existing pure query context, plus the exact current
function and instruction byte offset. These inputs do not constitute profile
or executable authority. My current C array step does not consult the
value-graph switch; I preserve that boundary rather than adding one.

I first require retained array types. Empty constructors query their exact
function/offset declaration through `ownership_queries.array_constructor`;
absent explicit selection uses the existing unambiguous tag query. Nonempty
literals derive the element descriptor from the first source-order operand,
resolve its retained array type and check every element's complete ownership
and structural nominal/array identity. I share that matcher with future
call/effect transfers instead of changing global stores' stricter layout test.

All other operations require the complete operand count before reading the
array type. Length accepts an unowned value or an accessible owner observation,
and refuses a staged owned array. Remaining operations refuse observation
escapes; only PUSH accepts an owned array. Extraction/disposal contracts remain
required for other operations on owned arrays. Index and slice-end operands
are exact nonowned, unobserved INT values. PUSH/SET check their element value
against the declared descriptor. GET/POP return the element type; other
operations return the array type. I derive result ownership from checked type
facts and retain the C step's error order, original256-slot limit and exact
messages. No new resource-array operation is admitted.

Success and refusal return independent complete frame snapshots, including
nested exclusion/local/global storage. My separate diagnostic fixture calls
the unchanged C step, keeps every old frame carrier/protocol and compares
complete reports against named independent expectations. I cover explicit,
absent, ambiguous and invalid constructor metadata; first/every literal
element; structurally equal and mismatched array recipes; owner/observation
flags; live/moved/held roots; index tags; stack bounds; malformed carriers; and
original source-order/atomicity controls. Old8/600 and new32-or8KiB/60 batching,
600-second builds and default10 shadows remain. Source routes, generated and
targeted actual-C sanitizers, paired hosts and complete composition remain
required before acceptance.

## My complete owner/reference-transfer contract

I port the fourteen current region, local/path borrow, reborrow, reference-field
and explicit owner move/store/pack/unpack instructions after their paired
local-state and frame primitives. I consume decoded operands and the existing
pure ownership query context. I retain the C switch's error order and strings,
including a failed path query before an exclusive observation guard, deeper
region/parent authority, exact field access before REF_SET stack typing, and
observation checks before moving or unpacking a local.

OWN_STORE accepts the C step's owned token and exact available destination,
then refines a known union arm before consuming it. A failed refinement returns
the original state. OWN_PACK preserves its distinct field rule: exact tag and
ownership, no observation, exact nominal record/union layout, with the C rule
for other field layouts. I share that rule with future aggregate construction,
without substituting the structurally checked array/effect matcher or global
storage's stricter descriptor comparison. OWN_UNPACK validates every field and
all output space before consuming the source and emits fields in declaration
order. I preserve the original256-slot bound and every local/path bound.

Every result owns mutable frame storage: local liveness/variants, reference
paths and control, globals and stack exclusion masks. Facts remain the existing
immutable shared metadata. I compare the unchanged C step using separate test
translation units: the owner/reference oracle and a read-only view of the
unchanged C state implementation. The view records mutable authority state
instead of exposing nondeterministic pointers or immutable reference counts.
It does not become product ABI. Original frame protocols and1001 reports stay
unchanged; state implementation source replaces only its ordinary object in
this diagnostic closure to avoid duplicate symbols.

Bounded instruction sequences test borrow/reborrow lifetimes, shared versus
exclusive parent access, region cleanup, live/moved/held roots, ownership
round trips, field order, exact destination/variant refusals, full-stack and
path boundaries, malformed carriers and direct caller isolation. Failed steps
normalize to independent input snapshots and can be followed by later probes.
Diagnostic-only empty-reference region setup covers the original UINT32_MAX
bound without executing billions of empty regions. Actual-C expectations,
complete source/generated-sanitizer routes, targeted C instrumentation and
paired qualification remain required. This source grants no function, graph,
effect, private-profile, executable or pure C-emission authority.

My [C expectation audit](evidence/darwin-current-source-20261124/diagnostics/affine-owners-c-audit-linux-20261125/README.md)
passes the unchanged1001 frame reports and14542 new complete owner/reference
reports in1067 actual C commands. Its independent reader checks36340 records,
mutable authority snapshots, exact source/tool/object guards and preserved
setup/path-fixture refusals. Strict C99 syntax and typed-AST checks pass; my new
NanoLang leaf has no executed-shadow or full-route acceptance yet. I keep this
work behind the already admitted foundation and prepared global/array gates.

## My aggregate-transfer contract

I port STRUCT_GET97, MATCH_TAG107, AGG_PACK251, AGG_GET252 and AGG_TAG254
as pure frame transfers. I retain the C step's distinct constructor contracts.
A copyable record requires value-graph mode, zero variant, a complete copyable
layout and exact declared field count, tags, physical layouts and absence of
owner/observation authority. A union constructor queries its retained ordinal
and variant before checking payload facts and the shared owned-field matcher.
Its output preserves the selected variant and derived ownership; it does not
acquire a new value-graph precondition.

MATCH_TAG checks a union stack value without observing or consuming it, scans
retained ordinals for the concrete layout/variant, and leaves exclusions and
selected arm unchanged. Branch refinement and convergence remain separate.
AGG_TAG accepts a plain union and returns a fresh INT. Ordinary record projection
requires a complete copyable layout and nonowned field. Union projection needs
a proven arm and permits only scalar/STRING fields; aggregate payloads still
require destructive unpack. I preserve the established observation fallback,
including its root/hold checks, scalar-first lookup and graph-gated STRING
lookup. That fallback does not impose an extra physical stack-tag check.

I compare complete frame and mutable authority snapshots with unchanged C,
preserve the1001 original reports, and use physical-tag/flag, declaration/layout,
variant, graph, full-stack, held-root and malformed-carrier controls. My new
source has useful shadows and caller-isolation assertions. C expectation audits,
executed NanoLang shadows, full source/generated-sanitizer/paired routes and
whole-walk integration remain distinct requirements. No AGG_SET implementation,
branch convergence, executable admission or pure emission follows from this
five-instruction leaf.

My [C expectation audit](evidence/darwin-current-source-20261124/diagnostics/affine-aggregates-c-audit-linux-20261125/README.md)
passes all1158 commands, original1001 unchanged reports and15222 complete
aggregate/authority reports. Its reader checks38061 records and preserves the
first5305 metadata-fixture refusals with unchanged expected reports. Strict
syntax, typed AST and shadow presence pass. I still require execution of my
new NanoLang shadows and all source/generated-sanitizer/paired routes before
whole-walk, profile or emission qualification.

## My foundation qualification failure

My [complete five-method terminal](evidence/darwin-current-source-20261124/diagnostics/affine-foundation-shadow-failure-linux-20261125/README.md)
atb43042ec finishes with four passes and one failure. Flow/FIFO, frame,
operand/profile and scalar retain189474 complete actual-C/five-route reports,
with all guards and original limits. The local method's C-seed native driver
build passes, but NanoVirt refuses its imported transfer shadow with BOOL_NOT
requires a boolean before any local differential route completes. I retain
all10409 local inputs and1001 original controls. I track localization in #1259,
do not accept the full foundation and do not admit the queued global/array gate.

## My effect-step and escape-identity contract

I port the current HANDLER_PUSH165 step as an isolated no-op, HANDLER_POP166
with inherited-floor bounds, and PERFORM167. The outer affine walk must still
validate handler signatures and destinations, create both handler edges and
bind activations; this leaf grants none of those obligations. I preserve the
existing frame activation, floor and handler order when popping local handlers.

I query the retained performed signature before checking exact positional
argument ownership with the shared effect/array matcher. I reject staged
owners/observations below the arguments, then unresolved local owners/borrows,
then failed escape recording, in the C step's order. The result always replaces
arguments with one fresh declared value, including VOID, with nominal or array
ownership derived from checked facts. Failed full-stack publication preserves
the original frame and escape table; no partial successor or summary is granted.

I represent the current function's bounded escape vector separately from its
frame. A matching handler catches an operation before the eight-function bound;
otherwise an existing equal key is retained before the256-key capacity guard.
New PERFORM entries use the signature's canonical key, including queries through
byte-identical aliases. Identity preserves C's equal stored lengths and strcmp
NUL-stop comparison; embedded-NUL suffixes matter only in private diagnostic
key comparisons because valid declared operation keys exclude NUL.

My key and escape C helpers require a nonnull checked module; I do not call
nvm_get_string with a null module in a diagnostic. Out-of-range string indices
are compared as absent keys, and defensive escape-table/function bounds remain
separate from whole-module admission. Diagnostic-only vectors can probe those
private defensive branches without authorizing them as a valid call graph.
Null-module PERFORM signature refusal and handler step behavior retain their
own defined C paths.

I compare unchanged C steps/identity helpers with independent complete frame,
mutable authority, handler context and escape-vector expectations, preserving
all1001 old frame/flow/FIFO controls. I require useful shadows, isolated returned
storage and the original full-source/generated-sanitizer/paired routes before
acceptance. This source does not resolve call summaries, handler/resume CFG,
private/executable admission or pure C-emission/cutover.

My [C expectation audit](evidence/darwin-current-source-20261124/diagnostics/affine-effects-c-audit-linux-20261125/README.md)
passes all878 commands, original1001 unchanged reports and8128 complete
frame/authority/key/escape reports. Its independent reader verifies11848
records, exact selected-source/production-C/tool/object guards, the proper
driver AST and shadow-presence checks, and the retained library-only missing-main
diagnostic. Executed NanoLang shadows and all source/generated-sanitizer/paired
routes remain pending. I do not infer handler CFG, call, profile, emission or
release acceptance from these C expectations.
