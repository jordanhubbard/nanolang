# I retain exact File and Result identities before execution

I record `task_21469c00ddb7a61dcb5ea5f4e9efd333` under72556/6931 on canonical
`c09c80af6768ce8fc5559fdfc9ed8806ae19b059`. PR825 qualifies only my private
File/OpenResult lifetime core; PR821 retains only service payload version1.
This contract refines the120-byte proposal in
[my full execution boundary](NANOISA_FILE_EXECUTION_BOUNDARY.md). It authorizes
no production until review, and no execution admission in this checkpoint.

## My audited dependencies

- `service_bindings.c/.h` implements exact56-byte v1. Its C struct, numeric
  statuses, overlap/atomicity contract and all existing v1 calls stay compatible.
- `service_bindings_module.c` validates five import identities/category signatures,
  not nominal layouts or per-outcome lifetimes. I extend version dispatch here,
  without treating successful metadata validation as host authority.
- `nvm_v2_layouts.c` already represents prior-order STRUCT/UNION fields and unit
  `TAG_VOID` alternatives. Its all-record forward extension cannot describe a
  forward-containing union table. I retain prior-order edges in this slice.
- `ownership_contracts.c:check_layouts` rejects every COMPLETE UNION; `descriptor`
  permits indexed TAG_STRUCT only. The shared single/batch authority queries
  call this validator. I do not weaken those checks to carry OpenResult.
- `nvm_v2_convert.c:conversion_ownership` calls the shared validator in both
  directions unless mixed admission succeeds. Required service claims already
  suppress mixed candidacy. A new explicit non-executing service-v2 transport
  branch is needed; neither generic validation nor mixed admission is a bypass.
- `nvm_v2_module.c` serialize and parse perform checked bridge conversion for
  ownership/service sections. I audit these indirect calls alongside direct
  in-memory attach/query/converter entry points; wire parsing alone is not enough.

I preserve ownership bytes and flags exactly. I never clear RESOURCE in a
synthetic module, reinterpret a resource union as an ordinary record, renumber
layouts, or publish shared ordinary authority from this descriptive plan.

## My version2 bytes and exact mapped layouts

I use separate private v2 raw types/functions, leaving the v1 raw API intact.
The v2 value owns five import indices and eight layout indices. Its120 bytes are:

| Byte range | Contents |
| --- | --- |
| 0..15 | u16 version2, u16 catalog1, u32 method_count5, u32 type_count8, u32 reserved0 |
| 16..55 | five pairs: u32 ordinal0..4 in order, u32 import index |
| 56..119 | eight pairs: u32 ordinal0..7 in order, u32 global layout index |

All integers are little-endian. Each index set is distinct and excludes
NO_INDEX. Raw decoding checks exact length before offsets, versions, counts,
ordinals and reserved bytes; it does not pretend to know module table bounds.
It stages output, permits input/output overlap, retains no input memory and
preserves outputs on failure. Encoding stages bytes before publication; size
storage must be disjoint, as in v1. Size-only mode validates the entire value.
Old readers reject v2 by version/length; neither v1 nor v2 becomes executable.

Module validation requires exactly five SERVICE imports, their existing exact
interface/method IDs and category signatures, no linked modules or callbacks,
and complete retained layouts plus ownership metadata. Required bits1(FFI),
7(retained layouts),8(ownership),9(service) remain necessary. Bit1 means imports
exist and grants no generic FFI dispatch. Feature/section contradictions refuse.
No known-feature assignment or import kind changes in this checkpoint.

I compare retained type names to full catalog type IDs and member/case names to
full catalog member IDs using stored lengths. These are checked interface facts,
not trusted compiler signatures. Source presentation names may differ; a later
publisher must emit these metadata IDs from checked declaration identity.
The immutable catalog supplies the actual IDs, order and scalar byte-domain
facts; I add a read-only catalog type accessor rather than duplicate that table.

| Catalog ordinal | Retained kind and exact ordered fields | Ownership flag |
| --- | --- | --- |
| 0 File | STRUCT, zero fields | COMPLETE\|RESOURCE (3) |
| 1 FileError | STRUCT: status/host_errno/cleanup_errno/bytes INT; eof/consumed/cleanup_failed BOOL | COMPLETE (1) |
| 2 ReadByte | STRUCT: value INT, eof BOOL | COMPLETE (1) |
| 3 OpenResult | UNION: Ok STRUCT→File, Error STRUCT→FileError | COMPLETE\|RESOURCE (3) |
| 4 WriteResult | UNION: Ok INT, Error STRUCT→FileError | COMPLETE (1) |
| 5 PositionResult | UNION: Ok VOID, Error STRUCT→FileError | COMPLETE (1) |
| 6 ReadResult | UNION: Ok STRUCT→ReadByte, Error STRUCT→FileError | COMPLETE (1) |
| 7 CloseResult | UNION: Ok VOID, Error STRUCT→FileError | COMPLETE (1) |

Scalar/VOID fields require NO_INDEX; record fields require the exact mapped
child, never a merely equal shape. All nested indices precede their containing
layout. Extra unmapped layouts may remain UNKNOWN (flag0) in this first private
transport; they must pass the existing structural decoder and prior-edge rule.
I refuse extra COMPLETE/resource authorities in this slice instead of inventing
composition with another executable profile. This is a bounded transport limit,
not permission to omit source declarations later. An expanded composition would
require its own review before public source acceptance.

The query publishes catalog→global, global→catalog/NO_INDEX, and per-kind source
ordinal↔global mappings without changing module indices. File is source STRUCT;
Results are source UNION. Runtime categories File/OpenResult/scalar Result are
explicit private enum facts, not COP TAG_OPAQUE, shared aggregate handles or
integer host tokens. Empty File's descriptor describes identity only; it does
not authorize OWN_PACK/AGG_PACK or a source literal.

## My private ownership transport validator

I parse ownership version1 completely on the service-v2-only path. This first
slice refuses ownership path-version2; File projections and borrowed subplaces
are not required to preserve these catalog bindings. Existing v1 service and
non-service ownership path behavior is unchanged.

I check layout count against the retained table, exact flags above, all padding,
function count, each local/parameter count, function result/parameter tag
agreement, every descriptor tag/mode/reserved/index and final byte consumption.
I bound tables by the existing retained maximum65,536 entries and checked input
length/products before allocation. Local/function counts use their actual wire
widths and checked module declarations. I do not use VOID as an unknown tag.

A descriptor with a mapped nominal index must have that exact layout kind's
STRUCT/UNION tag. A File/OpenResult descriptor must name its mapped layout; a
bare STRUCT/UNION descriptor with NO_INDEX supplies no nominal fact. Such
unresolved descriptors can be retained only as unresolved, never assigned a
catalog category. Scalar descriptors require NO_INDEX. Nonzero modes are legal
only on parameters, and only exclusive mode2 on exact File in this first slice;
all other descriptors are by value. These are syntactic identity/lifetime
requests, not verification of moves, initialized locals, branch refinement,
borrow exclusivity, public escape restrictions or execution safety.

I retain bytes in both bridge directions only after this full private query
succeeds. The query has no dependency on public verifier, converter or selector,
so validation cannot recurse. A transport decision returns a descriptive plan
or checked status, never an executable admission certificate. Shared
`nvm_ownership_contracts_validate` and authority queries remain unchanged and
continue rejecting these COMPLETE UNION facts. Copying ownership/layout bytes
into a module without its service section therefore cannot gain shared authority.

## My ownership, failure and consumer boundaries

The private query stages an owned plan and publishes it only after all checks;
invalid input/allocation failure preserves the caller's output. Checked size and
map allocations roll back completely. The plan owns every dynamic map and does
not borrow module buffers; immutable catalog views have process lifetime.
Getters check bounds and report UNKNOWN for unmapped identity.

Attach uses the already validated NlFilePlan plus exact v2 maps, stages bytes and
validates the candidate module before replacing nothing or publishing a new
owned payload. An identical existing attachment is idempotent; a conflicting
one refuses. Layout/ownership bytes are caller-established facts and are never
silently synthesized or modified by attach.

`from_nvm` retains its existing borrowed service/ownership payload lifetime and
cleanup-safe failure output, not an invented whole-output atomic-zero guarantee.
`to_nvm` owns independent copies and publishes only a complete module (failure
leaves NULL as today). I preserve every allocation-prefix cleanup and source
buffer lifetime requirement. Neither conversion executes instructions to derive
max_stack for a service module; unresolved execution facts are not fabricated.

All existing required-service guards stay before general/owned/mixed verifier
selection, VM direct and invocation entry, native/LLVM/Wasm conversion, linked
execution, FFI/COP initialization, reconstruction and metadata-dropping writers.
Malformed partial claims still refuse before output or dispatch. The v2 nominal
plan does not permit host operations, code execution, generic imported calls,
raw File construction, escaping public owners or selected-arm extraction.
Later flow/refinement, host grant, VM/native adapters and paired source/shadows
remain tasks72556/6931, not acceptance requirements silently declared complete here.

## My ordered checkpoints and qualification

1. Review this contract, then implement separate private raw v2 codec and staged
   nominal/ownership query. Send production before preparing/executing fixtures.
2. After that checkpoint qualifies, review the combined module version dispatch,
   converter retention and all unchanged consumer refusals before module gates.
3. Qualify golden raw bytes and overlap/atomicity; exact names/flags/case order,
   unit payloads, shapes, nominal aliases, permutations and disconnected UNKNOWN
   rows; every malformed count/length/reserved/mode/index/version; allocation
   prefix recovery and plan independence after source destruction.
4. Qualify in-memory↔container roundtrips preserving exact flags/bytes/maps,
   both bridge ownership lifetimes, feature/section/import matrix, direct query
   and public consumer refusal with sentinel outputs and no loader initialization.
   Include stripped service/forged mapping cases and unchanged v1/ordinary/owned/
   mixed profile controls. No service fixture executes bytecode or host File calls.
5. Freeze fresh Linux/Darwin C tools and source inventories, strict sanitizer and
   normal selected targets; retain first terminals. Run applicable module/provider
   closure gates if integration changes their sources. Report precise pins and
   do not turn descriptive roundtrip success into an execution claim.

### My first private implementation checkpoint

I keep the v1 codec and every converter/validator/selector untouched. Separate
`service_file_nominal.c` implements the fixed raw v2 value;
`service_file_nominal_plan.c` scans canonical prior-only layouts and ownership
v1 completely without allocating, then allocates one checked flexible map.
The preflight mirrors the applicable existing structural decoder rules, adds
exact stored-length catalog IDs and rejects trailing bytes. No decoded-layout
allocation error can be confused with malformed input on this private path.

Each function descriptor count is bounded by remaining bytes before local
iteration and compared to actual function arity/local/result declarations;
parameters require actual declared tags. Indexed descriptors must identify one
of the eight mapped layouts with its exact tag. Unmapped or bare nominal facts
remain UNKNOWN only through NO_INDEX and never acquire a runtime category.
Extra layout rows still have flag0. This first map owns no function-flow facts.
Plan getters publish only on success and preserve the caller's output on bounds
failure. Plan construction assumes input arrays remain valid and immutable
during the call; the published map is independent of them afterward.

I expose the immutable catalog type accessor used by the existing NlFilePlan;
I do not duplicate the eight type/member definitions. No fixture, build, host
operation or generated source execution accompanies this source checkpoint.
Provider-list/module integration remains in the later reviewed checkpoint.

My private query takes NvmModule, which has no serialized v2 feature-bit envelope.
Its acceptance checks payload/import/layout/ownership facts, not required wire
bits. The feature/section agreement matrix belongs to the later NvmV2 module
checkpoint. Prepared controls compare explicit and permuted nominal maps,
mutate every encoded layout/ownership byte, cover count/tag/mode identities,
raw overlap and output preservation, sole query allocation failure/recovery,
source destruction and old shared-authority/verifier/native-emitter/converter
refusal. Included-query interception and separately linked production both run;
old supporting objects retain their ordinary build flags, while the new codec,
query and catalog provider receive each selected sanitizer compiler. I add only
selected test recipes, not production provider/converter/selector changes.

### My next combined transport checkpoint

I preserve private qualification856620/720cf in its own trees and integrate
canonical829/e119 in a fresh retention tree. The only Make merge conflict is
additive selected targets; I preserve both exact blocks. Before code I identify
three intentional verdict changes for a fully valid service-v2 module:
`nvm_service_bindings_validate`, the NvmV2 metadata counterpart, and both bridge /
container roundtrips become transport positives. The existing fixture's v1 raw
decoder still rejects120 bytes; shared ownership/authority queries, general and
owned/mixed verification, VM/native/LLVM/Wasm and dropping consumers still
refuse. I will replace only those three old negative assertions with positive
retention/output/lifetime checks after combined source review; I preserve their
first private-gate evidence and all neighboring malformed refusals.

The NvmV2 metadata validator uses a bounded temporary execution-module *view*
of constants, signatures, imports and exact encoded layouts, then invokes the
already-qualified private query. It borrows scalar payload arrays, allocates only
checked adapter tables/layout bytes and the independent query result, and never
calls either bridge, public verifier or another module validator. Every allocation
is released before return. This prevents converter/query recursion. Full wire
feature/section consistency remains the existing container validation obligation.

An explicit service-v2 branch in conversion_ownership uses complete private
validation, preserves ownership bytes unchanged and records required ownership;
it does not route to mixed admission or the shared ordinary authority validator.
Service-v1 and all non-service paths preserve their old verdicts. From-Nvm skips
max-stack inference for every required service module and retains undeclared0;
metadata-only transport does not certify an executable stack bound.

I update all source/object/provider lists required by service validation's new
codec/query dependencies, including Make, module manifest and generated wrapper
link closure. Private test recipes filter the newly provided objects when they
compile those same sources directly; assertions are migrated only after source
review. No host grant, source generation, per-arm refinement or execution selector
is added. The new attach API stages exact v2 bytes, validates all candidate facts,
preserves every module field on failure and retains identical reattachment.

My combined fixture freeze migrates only version-aware metadata/bridge negatives
into positive exact-byte/map and borrowed/deep-copy lifetime controls. The new
module fixture covers direct NvmV2 table faults, required bits1/7/8/9, service /
ownership section corruption, attach idempotence/conflict/atomicity, all selected
consumer/CLI output refusals and unchanged loader state. Allocation prefixes
cover new adapter and private query plus retained-layout helpers and both bridge
directions, with recovery and sanitizer cleanup. I run old v1 transport and mixed
boundary controls separately. The inherited private fixture/gates remain sealed
in their original tree. No valid service program is executed by these controls.

### My first combined terminal and exact reader prerequisite

At frozen f5e07, Linux bootstrap passes251.959s, then the first combined GCC
gate stops5.492s at valid container deserialize, before allocation-prefix
controls. Static inspection finds the service-section reader still requires
exact56 bytes; the approved nominal writer emits120. I record `task_9664cf4ef91f101b7b2c1aed8e817a03` before
correction. I preserve the original source, wire, fixture executable and reports
without replay; the independent first Darwin run remains frozen too.

I will accept only exact56 or120 bytes at section framing, then retain complete
version/catalog/layout/ownership and feature consistency validation. Unknown
versions, cross-version lengths and malformed required sections still refuse.
The source delta receives review before fresh reader and dependent fixture
qualification. The prior successful bootstrap remains pinned to f5e07; a later
reader-only correction does not acquire a new bootstrap claim.

### My Darwin wrapper capacity prerequisite

I preserve the acc254 corrected Linux setup/GCC/Clang/adjacent passes and Darwin
focused transport pass (15.597s). Darwin adjacency stops at the embedded case of
`WrapperPublication.test_literal_paths_in_both_modes` (seven methods, one failure;
15.441s outer terminal). I record `task_ab82330ce20acb79e62cf7b8572bc6cf` before
correction. My fixed 16,384-byte object-list buffer holds 117 quoted paths; the
Darwin default temporary prefix plus the fixture's intentional alias needs about
16,733 bytes. This is a static capacity finding, not a replay of the failed binary.

I will build the complete object list with checked dynamic storage, preserving
every object, its order, and literal shell quoting. I will also build the link
command with checked dynamic fragments so the same closure does not encounter
another smaller fixed concatenation limit. The existing bounded compilation
command, trusted compiler command-fragment convention, private staging directory,
output checks and atomic rename remain unchanged. All allocated fragments are
freed on success and failure; allocation failure refuses without replacing output.
I do not shorten the path fixture or remove provider objects. Source review
precedes fresh Linux/Darwin wrapper controls and the remaining adjacent gates.
My passing service transport gates and original bootstrap claims retain their
own source pins; this prerequisite does not add service execution authority.
