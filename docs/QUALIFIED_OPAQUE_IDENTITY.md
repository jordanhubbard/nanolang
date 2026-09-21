# I bind opaque types to declarations across both producers

Task: task_2c7f68e4707b4d12b0ac8d29636fda63, under the open8bbc File/source parent.
This is a preimplementation contract. My SDK5bfd source and fixtures stay frozen;
no service execution, File lowering, or release acceptance follows from this plan.

## Observed boundary

Both5bfd installed runs completed clean bootstrap/install/reinstall and the first
three SDK methods. Their source-hidden C-seed program passed34shadows and executed.
Stage1 then refused the unchanged `Json.Json` annotation because the imported
function returns `Json`. I retain both failures and keep that exact fixture.

The C seed is not my identity oracle: `env_get_opaque_type` currently discards the
first qualifier and searches the global short name. An unknown qualifier can
therefore obtain an unrelated declaration. The Nano nominal prepass registers
records and unions in this branch but no opaque declarations; `types_equal`
compares their spellings. Its emitter recognizes a fixed opaque-name list rather
than all actual opaque declarations. I repair these authority boundaries together,
not by stripping qualifiers or treating all pointers as interchangeable types.

## Identity and visibility

My language identity is `(canonical declaring source, opaque kind, original name)`.
Repeated imports of one canonical source refer to one declaration. Same-name
opaque declarations in different physical files remain distinct. User module
labels, basenames, qualifier spelling and generated C names are not that identity.
The synthetic root parser used by direct unit callers has one explicit invocation
root owner; it cannot impersonate a loaded file owner.

A visibility row binds `(canonical importing source, original visible spelling)`
to exactly one declaration identity. I collect it from the existing actual import
AST and selected canonical target, including aliases/selective imports. I preserve
existing local-name precedence; conflicting imported bindings without a local
winner are refused rather than first/last-wins. Qualified lookup requires the
actual qualifier edge and actual opaque member in that target. Unknown qualifier,
missing member, wrong declaration kind and ambiguous unqualified lookup refuse.
No suffix-only fallback is allowed. Existing unqualified Json is an additive
positive control, not a replacement for Json.Json.

I retain the current opaque declaration syntax/visibility rule: it has no `pub`
field, and imported opaque types used by exported module signatures remain usable.
I introduce no new private/public grammar. An opaque and record/union with the same
name in one owner conflict in the shared type namespace. Across owners, kind is
retained and a record never becomes an opaque merely because spelling agrees.

## C producer boundaries

I extend the existing owned opaque registry and actual import namespace ownership,
not a new source scanner. `OpaqueTypeDef` retains original spelling, canonical
owner and an invocation-unique internal identity separately from its C ABI spelling.
A checked registration API returns failure, replacing unchecked growth/duplicate
suppression on this newly claimed path. The environment owns every copied string
and binding row; all new capacities and aggregate byte arithmetic are checked.
Failed registration/import preparation publishes no partial usable binding.

Actual root/module loading establishes an explicit canonical source context before
registration/typechecking and restores it on every return path. I do not reuse
`current_module`'s basename/module-label convention as physical identity. Import
namespace rows retain their importing and target canonical source independently
of existing display/module names. Repeated cached loads must use the same owner.
I inventory all environment create/clone/free and namespace registration callers
before editing these owning layouts; none may leave a copied identity borrowed.

`env_get_opaque_type` resolves a canonical identity or an authorized binding in the
current owner. It stops recursively discarding dotted prefixes. Complete opaque
facts copied into TypeInfo/signatures retain that identity. I audit annotation
resolution and copies in parameters, returns, lets, record/union payloads, arrays,
tuples, lists and nested callable types; equality uses declaration identity, not
ABI spelling. A syntax/name-resolution failure remains a refusal, not UNKNOWN
compatibility. Generated C retrieves ABI representation from the resolved row.

## Nano producer boundaries

I extend `compiler/nominal_bindings.nano` and the shared module binding map using
actual merged-file owners (`mb_owner`, `mb_source`, selected import target owner).
I preserve this branch's existing record and union registration/rewrite behavior.
The list/checker owner has a separate reviewed record path at0e72a33a6: this work
must retain its checked-record classification and must not admit opaque values as
record-list elements by name or layout guess.

Registration retains kind, original name, owner and internal target. Type namespace
lookup continues through the existing `#type:` key domain, but a parallel exact
kind/declaration fact distinguishes opaque from record/union targets. Generated
identifier allocation must be injective and collision-checked against original
symbols; key spelling alone is never declaration authority. Imports forward the
original declaration owner through `mb_add_target`, not the importer's owner.

I rewrite actual opaque declaration names and every existing annotation rewrite
site together, preserving original ABI spelling in the declaration registry.
`nb_scoped_type` keeps generic formal parameters lexical; qualified names resolve
only through installed visibility rows. Unknown qualified types refuse before
emission. Parameters parsed later from original token spans use the same owner map
as AST return/let/payload annotations. Constructor/clone/generated-list consumers
retain all newly needed facts; I do not silently normalize at an opaque data bridge.

The native C emitter consults the exact retained opaque registry in type parsing,
`type_to_c`, struct classification, array element classification and declaration
emission. Pointer representation does not erase nominal identity in the checker.
I retain existing fixed ABI/header mappings for known foreign opaque declarations;
custom opaque declarations need an explicit recorded pointer representation and
consistent generated declaration, never an inferred struct-by-value fallback.
If an existing foreign header's ABI representation cannot be established, I refuse
that unsupported representation rather than invent compatibility. This is not a
general foreign ABI or pointer-lifetime extension.

## Required controls and execution order

1. I inventory both producer resolution/emission/copy boundaries and send the full
   production checkpoint for review before fixtures execute. I coordinate shared
   typecheck/nominal files with the list owner; no competing registry is introduced.
2. I add real module cases for qualified Json.Json, unqualified Json, two aliases
   to one source, selective alias, nested importer-local aliases, repeated canonical
   imports and reversed order. Both different-directory same-basename and same-type
   spelling cases exercise the actual origin mapping.
3. I reject unknown qualifiers/members, local duplicate declarations, record/opaque
   kind mismatch, cross-origin opaque assignment/argument/return and ambiguous
   unqualified imports. Correctly qualified independent values remain usable.
   Same-origin aliases compare equal. Nested arrays/tuples/callables and generic
   formal shadowing retain exact identities; no test substitutes integer/pointer
   casts for typechecking. Unsupported aggregate/backend routes retain refusal.
4. I prove C registry allocation-prefix failure, source/AST lifetime copies and
   output preservation. Nano process-allocation failure is not claimed recoverable.
   Actual Cseed/Stage1/Stage2 select all added shadows; negative programs preserve
   output sentinels and do not silently publish generated programs.
5. I run fresh both-host compiler bootstrap after owning header/typecheck/emitter
   changes, then the complete unchanged SDK corpus, full File/provider/parser/
   schema neighbors and selected sanitizer scopes. I retain original5bfd terminals,
   exact source/tool/provider/product identities and full shadow selections.

Passing this dependency is necessary for installed paired-source acceptance. It
does not close full File source lowering, startup/shadow execution, cyclic/indirect
control flow, richer borrowing, or the overall5.1 release requirements.

## Resolution inventory and concrete representation

I found an existing C recursive nominal traversal in `src/nominal_types.c`.
`bind_nominal_records` already visits function signatures, lets, records, union
payloads, arrays/tuples/callables and expression children. I extend that traversal
for opaque facts; I do not create a second annotation walker. I audit its missing
complete record-field facts and opaque-specific TypeInfo slots explicitly.

| Boundary | Existing owner | Planned opaque change |
| --- | --- | --- |
| C registry creation/free | env.c, Environment | Checked staged opaque rows and import-binding rows; free every owned string |
| C source context | env_current_file, module loader cache | Copy canonical path into each row; synthetic root only when no source context was supplied |
| C imports | process_imports_owned's actual selected module_path and ASTImport | Stage exact importing-origin/visible-name/target-identity rows after successful dependency load |
| C annotation normalization | nominal_types.c | Rewrite authorized opaque references to immutable identity keys before signatures are registered |
| C checking | typechecker.c, checked expression/type facts | Compare resolved declarations for opaque assignment, argument and return, including nested annotations |
| C copying | env.c payload/signature copies, module.c metadata copies | Retain owned identity strings; no borrowed registry-entry pointer survives growth |
| C emission | transpiler.c opaque registry lookups | Resolve identity keys to existing pointer ABI representation, never emit a source path as an identifier |
| Nano registration/import | nominal_bindings.nano, module_bindings.nano | Add kind-aware opaque rows to the same actual-owner namespace; preserve records/unions |
| Nano annotation consumers | nb_rewrite, nb_type, token-based signature parsing | Normalize AST and later token-parsed annotations through one map |
| Nano checking/emission | typecheck.nano, transpiler.nano | Distinguish opaque facts from records and use their retained pointer representation |

My C owning-layout choice stays inside the approved registry/binding contract:
`OpaqueTypeDef` keeps its original `name` and separate `c_type_name`, and gains owned
`origin` and `identity` strings. An environment-owned opaque visibility array keeps
owned importing-origin, visible-spelling and target-identity strings. No opaque
lookup returns a borrowed pointer that a caller retains across registry growth.
I do not change AST/schema layouts or the unrelated record/union namespace ABI.
The actual import boundary stages all rows for one import before publishing them;
local declaration registration likewise validates all local duplicates/kind
collisions and stages its new rows before publishing. Failure leaves previous
rows intact and no partial new authority. Failed nominal AST normalization may
leave a rejected AST partially normalized; it cannot proceed to emission and is
freed normally. I do not claim whole-parser transactionality.

C identity strings use an unambiguous counted origin/name encoding with an opaque
kind discriminator; they are internal keys, not generated C identifiers. Registry
lookup recognizes only an exact registered key or an authorized current-owner
binding. Original short names remain separately available for diagnostics/ABI.
Each origin is copied from checked realpath of an actual supplied source context;
a NULL context has an explicit synthetic-root discriminator. A non-NULL missing
source cannot silently become the synthetic root. The existing env_current_file
borrow remains subject to its original lifetime; copied opaque rows do not borrow
that pointer, parser text, import AST strings or a module-cache entry.

Nano uses existing per-invocation merged-file owners and collision-free internal
identifier allocation, with exact kind/original/owner/ABI facts retained in the
nominal map. The original opaque AST spelling can remain source vocabulary;
normalized references use the internal target, and emitter lookup recovers the
opaque fact from that map. I will not add fields to the shared generated AST
schema merely to carry this compiler-owned registry. Normalized cross-producer
proof compares declaration tuples, not the different internal key encodings.

The C seed currently accepts broad struct/opaque arguments and opaque-to-int
legacy calls. I do not expand these conversions. Opaque-to-opaque checks must
compare identities instead of bypassing nominal checks; unchanged historical
scalar/foreign policies receive adjacent controls and remain separately scoped.
The native emitter's existing known opaque C spellings are preserved; a custom
actual opaque declaration uses the same pointer representation as the C seed,
with no record layout or pointer lifetime inferred from its name.

## Native identifier projection checkpoint

My static consumer audit found two additional required paths. C tuple typedef
reuse compares only coarse element tags, and generic-union symbol builders append
argument spellings through fixed 256/512-byte buffers. A declaration identity is
not a C identifier. I keep accepted generic opaque instantiations; I do not turn
this naming defect into a new source refusal.

I propose an emission-local owned name table for every complete derived name that
contains a registered opaque key. Its lookup key is a counted, tagged encoding
of the complete canonical TypeInfo/signature tree: base kind, nominal identity,
argument count and recursively framed arguments, array element, ordered tuple/row
components, callable parameters/result and their names/borrow facts. I do not use
the old underscore-concatenated spelling as a key: `Box<A_B,C,Opaque>` and
`Box<A,B_C,Opaque>` must remain distinct. Exact equal encoded byte strings share
one row; unequal strings receive distinct dense indices. The emitted spelling is `__nano_opaque_<prefix-index>_<row-index>`.
The table never establishes declaration authority: callers first resolve the
original opaque declarations, and the semantic registry retains origin/kind/name.
No ABI spelling or hash substitutes for nominal equality.

I reserve a prefix index against original source identifiers during my existing
recursive nominal traversal. I retain encountered canonical decimal prefix indices
from names beginning `__nano_opaque_<decimal>_`, without a second parser. I select
the first unused index after all modules/root declarations have been checked.
The selection is at most the number of distinct reserved indices, not a
user-selected enormous integer. Noncanonical decimal strings cannot equal my
output. I check function/parameter/type/enum/variant/field/let/loop/match names and
identifier/call references encountered by that traversal. Generated internal
families use different prefixes; foreign spellings explicitly present in source
are included. Arbitrary names/macros introduced solely by trusted C headers or
flags remain under the existing foreign build contract.

This adds an environment-owned checked array of reserved indices, released with
my new identity preparation state. It adds no AST/schema field. Native emission
owns its separate prefix/name rows until the completed output string has copied
their bytes. Each row owns the complete semantic key and short emitted spelling;
no borrowed array-element pointer survives growth. Failed allocation/overflow
refuses before output publication through the existing compiler failure boundary.
I do not claim a total compiler heap bound or recoverable Nano process OOM.

I audit every definition/reference path together: generic specialization names,
TypeInfo-to-native names, constructor/match/tag/payload names, signatures/locals,
to-string declarations/definitions, tuples and nested generics. Shared builders
first construct the complete checked semantic key in dynamic storage and then
project it; they cannot truncate a key before lookup or silently skip a required
definition. Ordinary names without opaque facts keep their existing spelling.
Tuple registry reuse compares complete retained annotations, and opaque tuple
fields use the existing pointer representation rather than a fabricated record.

My source-origin positive fixtures use different directories with the same base
filename and distinct explicit module declarations. I separately retain refusal
for duplicate default public module introspection labels. That existing label
policy does not collapse physical origins and is not expanded here.

Opaque-bearing generic registration must likewise retain a checked complete
TypeInfo copy and deduplicate by that tree, not the old flattened argument-name
array. I add a scoped full-tree registration path for these instantiations and
route their annotations, constructor contexts and match caches through the same
semantic key encoder. Existing ordinary registration remains unchanged. A copied
key/cache is still an internal semantic fact, never accepted as a source spelling
or a C symbol. All new registration allocations stage before publication and
have explicit rollback; no new whole-environment recoverable-OOM claim is made.

### Retained implementation work in progress

I have not qualified this source checkpoint. I retain the C complete-tree encoder,
checked annotation/signature copy and transactional generic registration in the
owned environment provider. My native emission table copies keys and names, and
projects derived generic, tuple and callable spellings before bounded C buffers.
My existing AST nominal walk reserves source name prefixes, including prefixed
spellings containing the generated family. I retain ordinary nonopaque names.

My independent Nano encoder frames the existing lexer's complete annotation
stream: each token kind and counted value, or a counted declaring-origin/original
opaque-name pair. Token punctuation preserves nested argument, tuple, borrow and
callable boundaries; insignificant lexical whitespace adds no identity. This is
an encoding of an already checked annotation, not a new authority parser. My
producer-specific internal keys need not have equal bytes; paired proof compares
actual origin, kind and original declaration facts. The Nano emitted family uses
the actual driver's source-token-checked binding prefix and a separate
`opaque_type_` suffix, with a per-emission key/name table. Process allocation
failure remains the existing Nano limit, not recoverable C MEMORY equivalence.

I have run strict C syntax checks only. Those checks found and corrected an
intermediate missing limits include and two joined-line indentation warnings;
no source program, compiler bootstrap, shadow or SDK gate ran for this work.
Callable/aggregate representation and every definition/reference consumer still
require the final source audit before fixture preparation and execution review.

### Derived declaration ordering: additional source inventory

My required tuple/callable generic audit found a concrete ordering dependency in
C: `generate_struct_and_union_definitions_ordered` runs before tuple/callable
registry creation, and `emit_native_type_info` currently handles nominal payloads
but falls back to a coarse type string for tuple and callable payloads. Merely
encoding their names cannot produce their required C definitions. I do not
introduce a new checker refusal for those accepted argument forms.

Before implementing this owning-layout extension, I propose the following exact
boundary for review. I prepare the existing function/tuple registries before
composite emission. I collect opaque-bearing derived annotations from complete
record fields, specialized union payloads, signatures and local annotations into
owned checked snapshots. Temporary substituted payloads are freed only after
these snapshots have copied their full facts. Each added derived row owns its
annotation and generated name; ordinary preexisting registry rows keep their
existing borrowed-AST lifetime. Destruction distinguishes these ownership modes.

I extend the existing declaration graph with tuple and callable rows required by
those opaque-bearing contexts. A by-value tuple/record/union component requires
its complete definition; a callback requires declarations of named parameter and
result types plus completed nested callable typedefs. Opaque and dynamic-array
pointers add no layout dependency. I distinguish declaration edges from complete
layout edges: legal callbacks involving their enclosing record must not become
false by-value cycles. When necessary, I forward-declare existing tagged record
and union names, then emit their bodies without repeating their typedef. A tuple
that needs a forward declaration receives a stable tag from the same checked
name table. This changes no field order, representation, function ABI or semantic
type identity. Truly impossible by-value recursive layouts cannot be repaired by
emitting an arbitrary fallback order; their existing source eligibility remains
separately explicit.

Every derived annotation emitted in a signature, tuple field, nominal payload,
constructor or local uses the same collected row. Failed allocation leaves a row
unpublished and the compiler emits no output artifact. Checked size/count
arithmetic covers added arrays and complete annotation copies; I do not promise a
total legacy compiler heap cap. Ordinary programs without opaque-bearing derived
types retain the old declaration order and names.

My Nano counterpart consumes its existing nominal dependency ordering and full
annotation strings. I audit its tuple/callback definition placement against the
same declaration-versus-layout distinction. This is a required paired boundary,
not a C-only replacement for the independent Nano producer.

Required controls include an opaque-bearing tuple payload, an opaque-bearing
callback payload, a nested callback with opaque parameter/result, repeated equal
annotations, distinct opaque owners, and a callback referring to its enclosing
record. They supplement the framed-name, failed-publication, same-basename module
and unchanged Json.Json controls. No execution is authorized by this additional
unreviewed layout proposal itself.

### Tuple child storage inventory and invariant

I retain complete `TYPE_TUPLE` children in the existing `type_params` vector;
I add no C TypeInfo field or metadata ABI layout. For this base kind only, a
complete vector has `type_param_count == tuple_element_count`, every child is
non-null, and each owned flat `tuple_types`/`tuple_type_names` entry reflects that
child's base/nominal view. A legacy tuple with zero child count and no vector uses
its existing flat entries. A nonzero unequal count or missing child is malformed,
not a generic-instantiation request. Empty tuples retain zero counts.

I inventoried every `type_params` reference in src C/headers/includes before
editing this representation. Parser/env copy/free, nominal recursion, metadata
pointer collection/serialization, C-backend kind checks and purity/resource walks
already visit child vectors without assuming generic arguments. Generic union,
HashMap/list and native monomorphization call sites require their base kind or
non-null generic name; I preserve those conditions. NanoVirt scalar/borrow
admission still rejects tuple shapes independently, so retaining children grants
no new bytecode authority. Reflection/docgen/LSP keep their existing explicitly
coarse tuple display behavior; display does not establish nominal identity.

The required changed consumers are full tuple equality and native keys (which
normalize a complete child against a legacy scalar/name view), concrete literal
checking/reduce comparison, tuple-index retained annotation, and generic payload
substitution. Substitution visits each full child and refreshes the owned flat
view, rather than leaving the original formal beside a concrete child. Metadata
round-trip must preserve both consistent views and all child graph edges.

Static inventory also found two native tuple-literal temporary TypeInfo objects
allocated without initializing fields outside the old flat tuple subset. Full
annotation equality requires zero initialization, explicit TYPE_TUPLE and an
explicit registry-owned temporary lifetime; I correct those paths rather than
reading uninitialized metadata. The C signature parser currently has an explicit
refusal of callback-valued parameters. I retain that documented unsupported-route
refusal; accepted callback returns and tuple/generic callback payloads keep their
complete signatures. This does not close the larger paired callable scope.

My literal consumer audit found another coarse lookup: the iterative C emitter
selects the first tuple typedef whose element tags match. I retain a checked
literal context in an environment-owned table keyed by the invocation's borrowed
AST pointer. Each row owns a complete annotation copy; repeated equal contexts
reuse the row, and conflicting concrete contexts refuse instead of replacing it.
I publish a row only after all strings/children/table storage succeed. Collector
and emitter use the same row; its key grants no namespace authority. Environment
cleanup frees the annotation and never dereferences its AST key. Original AST
lifetime is unchanged. Generic templates still require an exact concrete context
before using this table; I do not infer one from the literal's coarse tags.

The record parser requests a field TypeInfo with no separate callback-signature
output. I retain the accepted signature in that TypeInfo; callers requesting the
separate signature keep the existing ownership contract, and a caller requesting
neither output releases it. This adds no parser grammar or AST layout.

### Derived declaration source checkpoint (not qualification)

My C emitter stages `NativeDerivedGraph` before any composite body. Each derived
row owns its complete annotation, key and C name. Record/union rows borrow only
indices into the invocation's environment; substituted payload temporaries are
copied before release. Registry arrays may grow, but owned annotations and names
have separate stable allocations. I destroy signature/tuple registries before
freeing graph-owned annotations, and restore nested emission contexts. The old
registry's explicitly owned literal temporaries remain separately owned. I do
not transfer an AST or environment annotation into a registry destructor.

I forward-declare all graph record, union and tuple tags, then order complete
layout bodies. Callback parameter/result records and tuples require their
forward declarations, while a callback-valued result requires its typedef.
Record/tuple/payload values require complete layouts. An actual complete-layout
cycle fails before the driver publishes output. Non-generic union payloads and
function signature sidecars participate alongside record fields and concrete
substituted generic payloads. Both discovery paths refuse expansion beyond 128
edges instead of silently emitting an incomplete dependency inventory.

My independent Nano emitter stores complete annotation strings and derives the
same edge distinction in its existing nominal ordering. It collects actual
parsed, nominally rewritten ASTStruct fields rather than reconstructing those
fields with its incomplete late token scanner. The isolated scanner remains for
its existing callers and tests. Existing schema field C spellings and extern
record suppression remain explicit. Prototypes, global text and tuple discovery
are prepared before definition ordering, but their emitted locations stay the
same. Previously emitted derived names suppress duplicate typedefs. The active
forward-declaration mode is cleared after construction and reset per invocation.
Generic discovery visits tuple/callback children and substituted concrete union
payloads, not just the outer argument spellings. Compound array boxing/extraction
uses the same opaque-bearing C projection as its definition.

The complete tuple child vector reuses the existing C TypeInfo layout and
metadata graph fields. I change no schema or versioned ABI field. The SDK input
inventory includes the owning declaration-graph include. Nano process allocation
failure remains its existing runtime failure boundary; these arrays do not claim
recoverable OOM parity with my newly checked C registry publication. I promise no
total compiler heap bound. Legacy parser/environment/emitter allocation behavior
outside these named owning paths is unchanged.

Only strict C syntax checks and source/diff inspection have run for this source
checkpoint. I have not executed its Nano shadows, a compiler bootstrap, generated
C, or the installed SDK/File corpus. The forthcoming fixtures still require
actual C-seed/Stage1/Stage2 agreement, metadata round-trip and allocation-prefix
controls, nested callback/tuple/generic and enclosing-record cases, complete
array representation, both module-label boundaries and the unchanged Json.Json
installed-source case. All full File/source and release holds remain open.

### Array representation completion proposal

The last consumer audit found a representation decision beyond C identifier
projection. My C producer already stores record-like values with
`ELEM_STRUCT`/`dyn_array_push_struct` and exact value bytes. My Nano producer
currently stores records through heap-box pointers in integer slots; opaque
pointer leaves fall through without an explicit pointer/integer conversion, and
complete tuple/callback elements have no coherent matching path. Merely fixing
identifier spelling would not repair the accepted opaque-bearing array cases.

I propose the following bounded completion before implementing those consumers:

| Element with an opaque-bearing complete annotation | Value storage | Load |
| --- | --- | --- |
| Opaque declaration leaf | `ELEM_STRUCT`, copy `sizeof(void*)` from an addressable pointer value | Copy the pointer value out through its actual pointer C type |
| Tuple or generic-union value | `ELEM_STRUCT`, copy the complete generated value's `sizeof` bytes | Read the same complete generated type |
| Callback value | `ELEM_STRUCT`, copy the function-pointer typedef value's bytes | Read that exact callback typedef |
| Nested array | Existing `ELEM_ARRAY` pointer storage | Existing array pointer load, retaining the inner complete annotation |

This introduces no DynArray field or public runtime function. The existing
`size_t elem_size` ABI2 remains the owning runtime boundary. It does change the
Nano emitted storage tag for these concrete accepted element shapes, so I require
review of this choice before the consumer edits. A copied opaque pointer is
borrowed host identity; array disposal must not close/release its target. A copied
callback retains the existing callback escape/closure policy, not a new lifetime
authority. Tuple/union byte copies retain the existing non-owning native value-copy
semantics; this proposal does not invent a foreign destructor or generalized
aggregate ownership policy.

I will use one complete annotation classifier per producer for the affected
literal/empty literal, new/default, push, set, get/at, pop, iteration and supported
map/filter/reduce paths. Each store stages its source value once into addressable
storage and copies its exact width. Each load uses the matching retained full C
name, never a coarse element tag or a stripped qualifier. The C producer needs
retained complete array contexts where AST literals currently store only an
element tag; I will reuse the invocation-owned checked-expression context pattern
with complete copied annotations and explicit cleanup, not change the public AST
layout. Nano uses its already retained full annotation strings. Definition
collection must see these contexts before code emission.

I preserve ordinary nonopaque representations and unsupported-route diagnostics.
I require accepted opaque array cases through direct values, tuple/callback and
nested generic values, caller/callee and imported-module paths, nested arrays,
empty initialization followed by mutation, iteration and higher-order operations
already accepted by each frontend. Cross-producer runtime provider tests must
observe the exact tag/width and unchanged borrowed target identity. The original
SDK Json.Json tests remain unchanged. This proposal does not authorize execution
or close the broader File/SDK scope.

### Array consumer source checkpoint (unexecuted)

I extend the reviewed carrier choice to ordinary tuple and callback elements as
approved: an opaque-input map may return either, so preserving those accepted
compositions requires their complete value storage too. I keep ordinary scalar,
record, union and nested-array representations outside these affected paths.
C already uses width-aware struct storage for union values; Nano retains its
ordinary nonopaque nominal boxing. This checkpoint does not claim a new common
ABI for every ordinary nominal array.

I stage each affected store operand once. I copy loads with `memcpy` into typed
locals before a callback or loop body can mutate/reallocate the source. Callback
width comes from its actual generated typedef, not an assumed pointer width.
For fresh empty carriers I accept only tag STRUCT, width zero, length zero and
null storage, or the established exact width. Empty pop returns the existing
zero/default value without calling the runtime's width-asserting pop routine;
nonempty pop keeps the runtime assertions and success check. I add no runtime
field, destructor, borrowed-target release, or callback escape authority.

My C invocation owns a separate expression-to-complete-array snapshot table.
Rows borrow AST keys and own checked TypeInfo copies. I allocate copy and enlarged
table before publishing; failure sets the existing opaque-resolution failure
state. Destruction frees all snapshots independently of borrowed symbol rows.
Inferred tuple literals use the same checked tuple snapshot owner: actual child
facts, canonical opaque names and callable signatures are copied transactionally;
partial children are freed on failure. Tuple/callback literal validation retains
the initial indirect argument check plus recursive complete-child checks.

My Nano producer independently uses actual parsed children, canonical annotations
and declaration-backed callback signatures. I do not derive identity from a C
name. Its process allocation failure boundary remains unchanged; I make no
recoverable-OOM equivalence claim. Exact tuple/callback array emission requests
the derived-definition graph; ordinary standalone nonopaque annotations retain
the previous graph-selection rule. I retain supported variable iteration and
explicit unsupported non-range call iteration behavior for separate full-source
work, rather than claiming new iterable-expression support.

| Consumer | Retained boundary |
| --- | --- |
| Literal, empty literal, new/default | Complete expected or actual inferred element; typed staged copies |
| Push/set, get/at, pop | Exact annotation, STRUCT tag/width and typed addressable values |
| Variable iteration | Complete C checked loop symbol or Nano lexical annotation; copy before body |
| Map/filter | Actual full callback signature; complete output carrier; empty literal context from the callback |
| Reduce | C existing exact callback contract; Nano existing homogeneous scalar-only refusal remains |
| Slice/remove-at | Existing runtime byte-width behavior; complete source annotation retained for later consumers |
| Nested array | Existing ELEM_ARRAY pointer representation with retained complete inner annotation |

I require source fixtures for ordinary tuple/callback arrays, opaque maps to and
from those values, inferred literals, same-tag/different-owner rejection, empty
new/pop/mutation, slice/remove compositions, mutation during callbacks, and
cross-producer caller/callee tags and widths. I retain the original Json.Json
installed tests and all prior required identity, naming, layout, failure and
provider controls. Strict host-C syntax checks are the only compiler checks run
on this checkpoint; no bootstrap, Nano shadow, source program or SDK gate has
executed. Final production and fixture review still precedes fresh qualification.

### Absent exact-carrier load correction

Independent review of 8559c2273 found an unchecked NULL result from the existing
struct getter. I define an absent exact-carrier load as a terminal diagnostic and
`abort`, before any copy. I call the getter once with the already staged source
and index, retain that pointer only until the immediate typed copy, and then
invoke the callback or body. This includes a later snapshot-length iteration
whose source was shortened by an earlier callback/body. I neither invent a
zero/default value nor weaken the runtime width checks. Scalar getter behavior
and valid empty pop remain unchanged. Required subprocess controls cover negative
and upper-bound indices plus source shrinkage in variable iteration, map/filter
and each producer's supported reduce path, retaining the first terminal and
checking the explicit diagnostic rather than accepting an arbitrary crash.

### Resolved call ownership, spelling and discovery corrections

My native array selection now checks actual lexical/declaration ownership before
any intrinsic spelling. In C, a visible lexical value wins; an actual body,
extern declaration, source owner or alias row is distinct from a bodyless builtin
placeholder. Declared calls reach the existing visibility/argument checker and
ordinary staged native call path. Nano uses its lexical environment and resolved
parser function/extern rows, with the same precedence in checker, inference,
callee references and emission. I reuse the declared-call staging helper, retain
extern ABI names, and extend the existing private declared-array-push name family
to affected admitted declarations so runtime helper names cannot collide.

I do not widen the existing declaration-admission policy. C still rejects the
ordinary builtin declaration names it previously forbade, and its module checker
retains its separate registry policy. Permitted declarations, externs and lexical
callables must preserve their selected meaning; full cross-frontend builtin
policy remains the separately recorded follow-up. Required controls distinguish
admitted same-name calls from those unchanged declaration refusals.

My array comparison reuses the complete framed annotation-token encoder for
ordinary tuples/callbacks as well as opaque-bearing annotations. It compares
canonical rewritten nominal identities, not raw alias text or C symbols. Actual
alias resolution remains the nominal binding pass. Existing child-span splitters
trim their already separated annotations; callable registry reuse compares the
same semantic keys, retaining the exact-spelling fast path. I require whitespace
and real alias controls, including unequal child order and unequal owners.

I collect complete array leaf aliases from parsed lets/parameters and function
returns before the definition graph. Function bodies are prepared before this
assembly step; global text is also prepared before definitions. Empty literal
emission explicitly requests its element alias even with zero element expressions.
Affected global arrays, including nested arrays with an exact-carrier leaf, use
the existing ordered startup list instead of the old complex-literal zero
fallback. Their prepared text is emitted after typedefs/prototypes, so no
initializer executes during compilation. Source shadows check global-only tuple
and callback typedef placement, and actual qualification must prove initialization
and callable lifetime without changing the runtime grant boundary.

### My installed fixture checkpoint after 8a090

I add `tests/native_sdk_opaque_cases.py` through the existing installed SDK suite,
not a replacement runner. The original source-hidden `Json.Json` program and all
original clean-install, readonly, ABI, provider, concurrent invocation and owned
uninstall methods remain present. I run each additive source with the installed
C seed, Stage 1 and Stage 2 (`nanoc`) and retain the existing exact selected-shadow
multiset, generated C, compiler argv and input/product hashes. Source assets also
have a separate before/after byte, length and mode inventory.

My source corpus covers empty and populated global-only tuple/callback arrays;
ordinary tuple/callback operations and spacing; aliases to the same opaque owner;
distinct explicit labels in same-basename module files; reversed imports; nested
generic opaque instances and tuple/callback compositions; emission-prefix source
collisions; and actual C-provider observations of `ELEM_STRUCT` and `sizeof` each
value, including an actual callback typedef. These provider observations prove a
C ABI boundary, not a general foreign ownership or callback lifetime contract.
The full source contract's remaining allocation and cross-producer acceptance
obligations are not discharged by a provider observation alone.

I select only already-admitted main-scope declaration names for declaration
precedence positives. Every reviewed array operation also gets a lexical callable
with a tuple signature incompatible with the builtin's signature. Existing
reserved-declaration policy is unchanged. Copies survive callback-triggered source
reallocation; one source/index counter case checks evaluation exactly once.

My negative native executables have safe, meaningful shadows. Only the separately
supervised final executable requests a negative/upper-bound index or shrinks a
snapshot-length map/filter/iteration source. I require `SIGABRT` plus the exact
absent-element diagnostic; I reject sanitizer diagnostics as substitutes. A tiny
exec launcher disables core files while preserving the executable's signal status.
I do not run a null function pointer or copy a missing element. Compile refusals
preserve prior output bytes, require diagnostics and distinguish qualified-type
resolution errors from runtime absence. These fixtures are source-only at this
checkpoint: Python AST parsing and whitespace checks do not qualify any producer.

### General global-initializer integration after f39d

The two `f39d` producers pass C-seed/Stage1 build and smoke, then their generated
Stage2 shadow executable reaches `generate_fn_type_typedefs` with a NULL ordinary
array global. I preserve those actual terminals. I do not clear the mixed
`native_derived_emitted_names` set in `reset_fn_types`: that set also describes
emitted tuple/record layouts and has a separate lifetime.

I integrate root's reviewed source from `216ab627` and `2d89da158`. The C producer
collects top-level callable/tuple metadata before emitting declarations and uses
complete struct/union/enum/tuple/callable types for runtime-initialized globals.
Its existing direct scalar literal versus ordered runtime initializer selection
is unchanged. Nano now makes the same selection: direct number/float/string/bool
literal nodes may use static initialization; every other expression uses the
existing source-ordered startup path, including guarded mutable scalar values.
Empty and populated ordinary arrays therefore obtain actual runtime carriers.
No NULL-as-empty runtime rule or missing-initializer default is introduced.

I retain exact-array annotation discovery before layout emission, and globals are
still generated before mixed nominal definitions and callable typedefs are emitted.
The stronger opaque declaration/complete-layout graph remains authoritative.
Root's callable-global typedef-before-global assertion joins my existing tuple and
callback array ordering assertions without replacing any predicate. The old bit
and float classifier helpers and their shadows remain; the general selector now
covers those expression roots too.

Two additional installed source cases join all original controls. Ordinary
`array<string>`/`array<int>` globals are read without any reset call, and a mutable
empty carrier is pushed/popped without replacing it. A separate nonliteral case
checks dependency-ordered references and calls, mutable scalar startup, record,
tuple, array, callable and union values, computed boolean/string/integer roots.
These additions audit global initialization; they do not replace root's separate
written aggregate-field ordering acceptance or merge PR937. No corrected build,
bootstrap, shadow or installed-source execution has run at this checkpoint.
