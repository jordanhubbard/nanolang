# RFC: Row-Polymorphic Records

- RFC PR: not submitted
- Status: Postponed
- Author: Rocky / Natasha
- Created: 2026-03-31
- Decision: 2026-09-08
- Task: `task_a39aac00600aa77b55ad92ac70a2d1bf`

## Decision

I postpone this proposal. I have a C-seed inference prototype, but I do not
have enough evidence to accept a language feature. The self-hosted frontend
does not implement the syntax or semantics, the C backend emits stubs for open
record parameters, and NanoISA does not preserve a record-row contract through
serialization. Agent-message schemas are a useful workload. They are not a
substitute for frontend compatibility, a stable ABI, or measured performance.

This decision completes the 5.0 RFC obligation. It does not put
row-polymorphic records in the 5.0 language contract. Existing code is a proof
of concept and must not be described as a portable feature.

## Summary

The proposal adds structural record types whose named fields are required and
whose row tail permits additional fields. A function requiring a field could
then accept nominal records with different declarations, provided their field
types agree.

## Canonical Syntax Candidate

If I reconsider this RFC, the only syntax under consideration is:

```nano
fn name_of(value: { name: string | rest }) -> string {
    return value.name
}

shadow name_of {
    assert (== (name_of { name: "Ada", role: "compiler" }) "Ada")
}
```

The grammar candidate is:

```text
open-record-type := "{" field ("," field)* "|" row-name "}"
field            := name ":" type
row-name         := lower-case identifier
```

The row separator is `|`; `..` and `...` are not alternate type spellings.
Record construction remains ordinary nominal record construction. This RFC
does not accept anonymous record literals, record spread, or record patterns.
Those are separate language changes and need separate evidence.

## Semantic Contract Required For Reconsideration

An accepted revision must define all of these rules without relying on a
backend representation:

- A row variable ranges over a finite map from field names to types.
- Field order is semantically irrelevant, and duplicate labels are errors.
- A value satisfies `{ f: T | r }` only when it has `f` with exactly the type
  `T`; mutable fields are invariant.
- Width compatibility applies when an open record is expected. Closed nominal
  records do not become structurally interchangeable with each other.
- Each call instantiates a generalized row variable freshly. Escaping or
  ambiguous row variables are errors.
- Inference must report the required field, actual field type, expected field
  type, and both source spans when unification fails.
- A missing required field and a duplicate field are distinct diagnostics.

An accepted revision must also state whether row variables interact with
generics, unions, mutable fields, resources, patterns, and module boundaries.
Silence on one of those cases means the construct is rejected there.

## One-IR Contract Required For Reconsideration

Rows are compile-time evidence, not dynamic maps. Lowering must erase a row
variable only after resolving each required field to a concrete nominal record
and field index. NanoISA must carry enough verified metadata to reject a
consumer whose required field name, type, mutability, or layout does not match.

The `.nvm` encoding must be versioned and deterministic. Serialization followed
by deserialization must preserve the nominal record identity, required-field
set, field types, and resolved field indices. NanoVM and `nvm2c` must consume
the same instructions and metadata; neither frontend may communicate a hidden
layout convention directly to one backend.

No boxed map, fat pointer, or tagged dynamic-record ABI is accepted by this
RFC. A future proposal may choose one, but it must provide compatibility rules
for separate compilation and measurements against direct nominal-field access.

## Acceptance Corpus Required For Reconsideration

The same source corpus must run through the C seed and self-hosted frontends.
Each accepted case must compare canonical NanoISA after a `.nvm` round trip,
then compare observable results in NanoVM and AOT C. Each rejected case must
compare structured diagnostic codes and relevant spans.

The corpus must include:

- one required field and several extra fields;
- reordered fields;
- nested records and generic field types;
- independently instantiated row variables at two calls;
- missing, duplicate, and wrong-typed required fields;
- mutable-field variance rejection;
- module-boundary and separately compiled records;
- deterministic serialization of equivalent inputs;
- VM/AOT equivalence after serialization; and
- compile-time, module-size, and runtime field-access measurements against the
  equivalent nominal-record program.

Until that corpus exists, I distinguish a C inference experiment from a
language feature. The distinction is useful precisely because the experiment
already exists.

## Motivation

Message routers often need a small stable header while producers carry
different payload fields. Structural requirements can express that shape
without forcing every producer into one large nominal declaration. The same
need appears outside agent messaging in adapters and schema evolution, so any
accepted design must stand on general language semantics.

## Drawbacks

Row inference adds a second namespace of variables, more complex diagnostics,
and constraints that must survive module boundaries. Structural compatibility
also complicates ABI evolution: two records that look compatible in source can
have different physical layouts. Erasing that distinction too early produces
backend-specific behavior.

## Alternatives

- Define a nominal header record and pass payloads separately.
- Use explicit accessor traits or functions over nominal records.
- Use a tagged union for the closed set of message variants.
- Decode a dynamic map at the protocol boundary and convert it to nominal
  records before typed code uses it.

These alternatives are more explicit and already fit a nominal ABI. They are
the supported choices while this RFC is postponed.

## Reopening Criteria

A new RFC may reopen this decision when it supplies the semantic and One-IR
contracts above, the dual-frontend conformance corpus, `.nvm` round-trip
evidence, VM/AOT equivalence, and performance measurements. Reopening requires
a new discussion and final comment period; this postponed document is not an
advance acceptance.
