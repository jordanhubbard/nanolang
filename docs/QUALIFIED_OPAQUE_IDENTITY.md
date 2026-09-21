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
