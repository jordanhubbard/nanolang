# My dotted union constructor boundary

I retain the original native-discovery refusal at b33 before C emission under
task_e0fbb45b12b64012b39a632f5c90d21e. The source declares Held<T>.Items and
initializes a declared Held<Item> from Held.Items{values:xs}. The parser can
represent this spelling as AST_STRUCT_LITERAL. My imported qualified nominal
binder applies the annotation-slot rule too early and rejects the constructor
before the existing checker can select its union variant.

I classify only this actual literal node. A dotted name in a type annotation is
still a qualified annotation; I do not excuse arbitrary dotted names. I split
the final constructor component and resolve its prefix from the complete local
union declarations first, then the existing owner-aware imported union identity.
Local type declarations keep precedence; ambiguous or cross-kind declarations
do not turn an unresolved name into authority. An imported namespace must belong
to the current owner and export the selected union. The selected variant must
occur exactly once in that exact declaration. No global unique-spelling fallback
is permitted. Normal qualified record literals continue through nominal_slot.

Once I have that exact constructor, I prepare independently owned prefix and
variant strings before changing the AST. I refuse a union spread literal. On
allocation or resolution failure the original node and every owned child remain
unchanged. On success I transfer its field names and values once into the existing
AST_UNION_CONSTRUCT representation, set type_info to NULL for the existing
contextual-instantiation path, clear the old union storage, publish the new tag,
and release only the replaced old struct-name string. Existing complete field,
arity, owner-aware destination and generic-substitution checks remain required.
This is parser representation normalization, not a new enum/union ABI.

I retain the original discovery source rather than rewriting its constructor.
I add actual parser/checker controls for local constructors, explicit generic
constructors, forward declarations, imported alias ownership, wrong or unknown
variants, inaccessible/wrong-owner prefixes and ordinary qualified records.
Checked allocation-prefix controls must preserve original AST ownership on
failed conversion and admit fresh independent recovery. The original18 source
and8 native methods, both hosts and full bootstrap remain required. No source
program or allocation-failure control runs before review of the complete source
and fixture checkpoint.

## My fixture checkpoint

I include the actual nominal binder in a separate allocator-hook translation
unit, renaming only its public entry to preserve the real provider too. Local
and imported successful conversions each measure two string allocations. I
refuse every measured allocation prefix in persistent and transient modes,
compare the complete unchanged AST and child aliases, and then perform fresh
independent recovery. Successful strings transfer to the ordinary AST destructor;
I do not claim that destructor belongs to the hook domain.

My parser/checker cases retain both dotted and explicit generic syntax, both
forward and earlier declarations, and wrong variant/payload refusals. Separate
parsed binder controls retain exact exported alias ownership and ordinary
qualified record behavior. Inaccessible, wrong-owner, unknown-alias, spread,
ambiguous-kind and duplicate-variant controls preserve the original node. I add
these controls to the existing identity method without removing any of my
original eighteen source or eight native methods. This checkpoint is source-only
until review and fresh corrected qualification.

Before execution I corrected three whole-object fixture snapshots to use
`memcpy` rather than struct assignment. I compare object representations with
`memcmp`, so the snapshot must preserve padding bytes too. This changes my
evidence precision, not production behavior or any refusal assertion.


## My imported parser prerequisite

I retain the 51e Linux first focused assertion: my parsed alias.Box.Value
control is a field-access node, not a named literal. My current primary parser
recognizes a brace only after one or two name components. The third component
falls through to ordinary field access and the brace can become a separate
statement. Puck completed fresh providers but did not replay this assertion.

I extend the existing named brace-literal route to the exact three-component
namespace.union.variant form. I look ahead for identifier/dot/identifier/dot/
identifier followed immediately by the brace, retain the existing uppercase
last-component convention, and preserve all one/two-component and ordinary
field-access routes. I allocate the complete dotted name before consuming its
parts; failed allocation reports a parser error without partial name ownership.
The existing field parser/destructor owns the resulting single literal. Only
my owner/export/variant binder decides whether it is a declared constructor.

I retain the original alias source and add exact body statement counts, field
vector contents and full parser-consumption controls, including plain chained
field access without braces, ordinary qualified records and malformed missing
components/braces. These are parser checks only; actual imported checker/native
acceptance remains independently required. My two-allocation binder sweep does
not claim to instrument this new parser name allocation.
