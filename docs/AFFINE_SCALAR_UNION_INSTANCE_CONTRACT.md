# My exact affine scalar-union contract

I finish `task_a18a9f752536469faafc4d3ebec01dfd` on the canonical match behavior
qualified in PR889. I do not replace that parser, checker or ordinary NanoISA
emitter. This contract supersedes the representation assumptions in recovered
checkpoint `a9ea7d69bff89ca413e179b587b6bdb549d0a857`.

## I keep concrete identities distinct

Each concrete union instance has an internal nominal key made from the resolved
source declaration and module identity plus the recursively resolved identities
of every type argument. For example, `Choice<int,string>` and
`Choice<float,bool>` may appear in the same module and receive different
retained layout indices. Equal unqualified names from different modules do not
compare equal, and aliases do not create a second identity for the same resolved
instance. I deduplicate only equal resolved keys. A source declaration index,
display spelling or first encounter is not a runtime instance identity.

I retain a canonical concrete spelling as an advisory layout name for dumps and
diagnostics, never as the authority for equality. The initial admitted affine
profile may derive its key from a checked root-module declaration because it
continues to refuse imported union instances; unresolved declaration provenance,
aliases or cross-module identity must refuse rather than fall back to text. I
substitute every generic parameter before classifying payloads. Unresolved
parameters, resource-bearing payloads, nested aggregates and unsupported scalar
kinds refuse before module publication. I do not silently select the first
concrete instance encountered.

## I retain exact variant shapes

The ordinary retained layout holds the concrete instance's payload fields in
variant-major order. Versioned ownership metadata supplies the information the
layout table does not contain:

```text
concrete layout index
variant count
for each source-order variant:
    retained variant name
    payload offset
    exact payload arity
```

The offsets form a complete, non-overlapping partition of the retained payload
fields. Empty variants have an exact zero-length slice. Every payload field has
its substituted scalar tag and source field name. This permits heterogeneous
variants such as `IntValue { value: int }` and
`TextPair { left: string, right: bool }` without pretending that equal field
positions share one type.

I extend the ownership payload with a new version rather than changing the
meaning of version 1 or 2 bytes. Older modules keep their current validation.
The new table is bounded, canonical and consumed through one checked query.
Malformed counts, offsets, names, duplicate layouts, missing concrete layouts
and trailing bytes all refuse without partially publishing facts.

## I share one version 3 extension envelope

Version 3 is not a union-specific grammar. It is the single shared ownership
extension envelope used by union-variant facts and ordinary record array-element
authority. I retain the version-1 layout flag and function descriptor prefix
byte for byte. The remaining version-3 bytes are:

```text
path_bytes: u32
paths[path_bytes]
extension_count: u32
repeat extension_count times, in strictly increasing kind order:
    kind: u16
    revision: u16
    payload_bytes: u32
    payload[payload_bytes]
    zero padding to four-byte alignment
```

`paths` contains the complete existing version-2 path suffix: its `path_count`
and all path rows with their existing row alignment. Even no paths therefore
uses four bytes containing a zero count. `path_bytes` is at least four, is a
multiple of four and bounds a subcursor whose exact end is checked. Version 2
does not gain `path_bytes` or extensions: its path suffix still starts directly
after the function descriptors and must consume the whole ownership payload.
Version 1 still ends after its function descriptors.

Extension kind 1 is `UNION_VARIANTS`, revision 1. Its payload is the exact
union table already required here: `union_count:u32`, then one row per retained
union in increasing layout-index order. A row is `layout:u32`,
`variant_count:u16`, zero `reserved:u16`, followed by source-order variant rows
of `name_idx:u32`, `field_offset:u16`, `field_count:u16`. The payload must end
exactly after the last row and must cover every retained union layout once.
Within a validated module, `(module artifact identity, layout index)` is the
runtime nominal identity. The producer's resolved declaration/module/type-arg
key decides that layout index; the retained name remains advisory.

Extension kind 2 is `ARRAY_FIELDS`, revision 1, owned by the ordinary
record-array authority contract. It carries that contract's counted eight-byte
type rows and twelve-byte layout/field bindings. Both extensions may coexist.
The common validator validates the complete path suffix, every extension and
all cross-references before any feature-specific query returns facts. Kinds are
unique and mandatory-understanding: an unknown kind or revision refuses rather
than being skipped. A version-3 producer emits at least one extension and uses
the lowest version that represents its facts.

The earlier `NVM_OWNERSHIP_UNION_VERSION` name and direct path-plus-union suffix
were provisional PR893 implementation, not an accepted wire contract. My
shared reader and producer now use the envelope above. Until the independent
`ARRAY_FIELDS` validator lands, its recognized kind still refuses as mandatory
but unsupported; I do not project the union half of a module I cannot completely
validate. I do not publish two version-3 grammars or claim compatibility with
the discarded draft-only version-3 bytes.

## I prove the selected variant before projection

A constructor must agree on concrete layout, variant membership, exact arity
and every substituted payload tag. The affine stack retains both concrete
layout and selected variant for a newly constructed value.

For an unknown parameter or call result, a match edge establishes the selected
variant before any payload projection. The affine match emitter uses the
existing `MATCH_TAG` control-flow instruction for this proof; it does not infer
a variant from a field position. The true edge refines the checked union value,
while the false edge retains only the exclusions needed for source-order
dispatch. A payload projection requires a proven variant and checks its exact
slice. The runtime carrier still checks tag, concrete layout, variant and field
bounds as defense in depth.

That refinement belongs to the exact tested value and control-flow edge. A
later projection through another local, parameter or expression cannot inherit
it merely because the static union instance matches. A join that cannot prove
the same selected variant on every incoming edge drops the refinement; a later
projection must establish it again. Producer metadata and runtime checks must
identify both the concrete instance and the refined value rather than treating
`MATCH_TAG` as a module-wide fact.

I preserve once-only scrutinee evaluation, lexical payload bindings,
source-order first success, exact `bool` guards, guard side effects, enclosing
function `return`, final-expression match values, owner state at joins and the
terminal unmatched invariant already established by PR889. A false guard
continues to the next source arm, including another arm for the same variant.

## I require all routes to agree

The structural verifier, affine analysis, NanoVM and generated native C must
read the same concrete identity and variant facts. Native carriers own their
payload exactly once and release every losing, returned and trapped path.
Calls, locals, results and control-flow joins compare the concrete layout;
variant refinement is path-local and cannot leak through an incompatible join.

Statement and value-producing matches both qualify. Value arms use their final
expression as the match result; `return` always exits the enclosing function.
No route invents a default zero, void value or last-wildcard behavior.

## My acceptance

Before admission I require focused malformed-metadata and allocation-failure
controls, simultaneous generic instances, heterogeneous and empty variants,
constructor arity/tag refusals, call/local/result transport, statement and
value matches, repeated guarded arms, once-only effectful scrutinees, owner
joins and prior-output preservation. Identity controls include equal
unqualified names from distinct modules and alias resolution without textual
conflation. Refinement controls include a `MATCH_TAG` on one value followed by
projection from a different receiver, and a join that loses the selected
variant; both projections must refuse until independently proved. The same
unchanged sources run through C-seed, Stage 1, Stage 2, verified NanoVM and
strict generated native C.

I then require fresh bootstrap and the complete adjacent affine state,
bytecode and owned-value gates on Linux and Darwin. The separate Darwin
self-host stack prerequisite may supply its reviewed default compilation
policy after merge; it is not scalar-union evidence. No bounded child closes
PR522 or authorizes a tag or release.
