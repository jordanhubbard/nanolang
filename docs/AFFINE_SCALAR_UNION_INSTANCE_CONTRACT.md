# My exact affine scalar-union contract

I finish `task_a18a9f752536469faafc4d3ebec01dfd` on the canonical match behavior
qualified in PR889. I do not replace that parser, checker or ordinary NanoISA
emitter. This contract supersedes the representation assumptions in recovered
checkpoint `a9ea7d69bff89ca413e179b587b6bdb549d0a857`.

## I keep concrete identities distinct

Each concrete union spelling is a separate nominal runtime identity. For
example, `Choice<int,string>` and `Choice<float,bool>` may appear in the same
module and receive different retained layout indices. I deduplicate only the
same canonical concrete spelling. A source declaration index is not a runtime
instance index.

I retain the concrete spelling as the layout name. I substitute every generic
parameter before classifying payloads. Unresolved parameters, resource-bearing
payloads, nested aggregates and unsupported scalar kinds refuse before module
publication. I do not silently select the first concrete instance encountered.

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
joins and prior-output preservation. The same unchanged sources run through
C-seed, Stage 1, Stage 2, verified NanoVM and strict generated native C.

I then require fresh bootstrap and the complete adjacent affine state,
bytecode and owned-value gates on Linux and Darwin. The separate Darwin
self-host stack prerequisite may supply its reviewed default compilation
policy after merge; it is not scalar-union evidence. No bounded child closes
PR522 or authorizes a tag or release.
