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
five routes. The combined permanent fixture now includes both groups; paired
Darwin acceptance and original module/claim regressions remain pending.
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
boundary grants no stack, type, lifetime, effect or host authority. Full paired
producer/platform qualification remains pending.
