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
