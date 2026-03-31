# Row-Polymorphic Records Design

**Status:** Implemented (see `src/type_infer.c`, `src/type_infer.h`)  
**Author:** nanolang type system  
**Feature branch:** `main`

---

## Overview

Row-polymorphic records extend nanolang's Hindley-Milner type inferencer with
*open record types* — record types that can be unified against any record
that has *at least* the specified fields, plus an unconstrained *row variable*
for the rest.

This is the standard encoding from the literature (Wand 1987, Rémy 1989,
Leijen 2005) adapted to nanolang's existing HM type framework.

### Motivating Example

```nano
# A function that works on any record with a "name" field of type String,
# regardless of other fields.
fn greet(person: {name: String | r}) -> String {
    return (format "Hello, %s" person.name)
}

# Works with a minimal record:
let alice: {name: String} = {name: "Alice"}

# Also works with a richer record:
let bob: {name: String, age: Int} = {name: "Bob", age: 30}

(greet alice)   # ✓ String
(greet bob)     # ✓ String — r unifies with {age: Int}
```

---

## Type Representation

### Syntax

| Form | Meaning |
|------|---------|
| `{name: String, age: Int}` | Closed record — exactly these two fields |
| `{name: String \| r}` | Open record — has `name: String`, plus any fields bound by row variable `r` |
| `{name: String \| {}}` | Open record with empty tail (equivalent to closed) |
| `{ ...base, extra: Bool }` | Spread expression — copy all fields from `base`, add `extra` |

### Internal HM Representation

A new HMKind variant `HM_RECORD` is added alongside `HM_VAR`, `HM_CON`, `HM_ARROW`:

```c
typedef enum {
    HM_VAR,       /* type variable α */
    HM_CON,       /* concrete type: int, float, bool, string, void */
    HM_ARROW,     /* function type: τ1 -[ε]→ τ2 */
    HM_RECORD,    /* record type: { f1:τ1, …, fn:τn | ρ } */
} HMKind;
```

The `HM_RECORD` node carries:

```c
struct {
    char    **field_names;    /* sorted field labels */
    HMType  **field_types;   /* corresponding field types */
    int       field_count;
    HMType   *row_tail;      /* NULL = closed; HM_VAR = open (row variable) */
} record;
```

A **row variable** is simply a regular `HM_VAR` used as `row_tail`.  The
substitution machinery handles it transparently.

---

## Unification Algorithm

Row unification follows the *row-merge* strategy: to unify two record types,

1. **Collect all field labels** from both sides.
2. For each label present in both: unify the corresponding field types.
3. For labels present only on the left: add them to the right's row tail
   constraint (or fail if right is closed).
4. For labels present only on the right: add them to the left's row tail
   constraint (or fail if left is closed).
5. **Tails:**
   - Both closed (`NULL`): succeed if field sets match exactly.
   - One open, one closed: the open tail must unify with the set of
     extra fields (or empty if none).
   - Both open: unify tails with a fresh row variable representing
     the combined remaining fields.

This is the "lacks" / "has" predicate encoding in practice.

### Occurs Check

The occurs check is extended to walk into `HM_RECORD` nodes, checking both
field types and the row tail, to prevent infinite record types.

---

## Record Spread Expressions

A spread expression copies all fields of an existing record and adds or
overrides fields:

```nano
let extra: {name: String, age: Int, role: String} =
    { ...alice, age: 25, role: "engineer" }
```

At the type level, spreading record `r: {f1:τ1 … fn:τn | ρ}` and adding
fields `{g1:σ1 … gm:σm}` produces:

```
{f1:τ1, …, fn:τn, g1:σ1, …, gm:σm | ρ}
```

If any `gi` collides with an `fi`, the new type `σi` overrides (shadowing).

---

## Open/Closed Record Patterns

Pattern matching distinguishes open and closed record patterns:

```nano
# Closed pattern — matches only records with exactly these fields
match msg {
    {kind: "ping"} => (handle_ping)
    {kind: "pong"} => (handle_pong)
}

# Open pattern — matches any record that has a "kind" field
match msg {
    {kind: "ping" | _} => (handle_ping)
    {kind: k       | _} => (handle_unknown k)
}
```

The `| _` tail discards remaining fields; `| r` binds the remainder.

---

## agentOS IPC Schema Examples

Row-polymorphic records are particularly useful for agentOS IPC, where
messages share a common envelope but carry variant payloads.

### Common Envelope

```nano
# Every IPC message has at least these fields:
type IPCEnvelope<r> = {
    id:      String,
    ts:      Int,
    version: Int
    | r
}

# Specific message types extend the envelope:
type SpawnRequest = IPCEnvelope<{
    op:      String,   # "spawn"
    image:   String,
    args:    String
}>

type QueryRequest = IPCEnvelope<{
    op:    String,   # "query"
    sql:   String,
    limit: Int
}>
```

### Router Function

```nano
# Routes any IPC message to the appropriate handler.
# The row variable means this works for ALL envelope variants.
fn route_ipc(msg: {id: String, op: String | r}) -> void {
    match msg.op {
        "spawn" => (handle_spawn msg)
        "query" => (handle_query msg)
        _       => (handle_unknown msg.id msg.op)
    }
}
```

### Schema Narrowing

```nano
# Narrow from envelope to concrete type when op is known:
fn handle_spawn(msg: {id: String, op: String, image: String, args: String | r}) -> void {
    (spawn_agent msg.image msg.args)
    (send_ack msg.id)
}
```

---

## Implementation Notes

### Files Changed

| File | Change |
|------|--------|
| `src/type_infer.h` | Added `HM_RECORD` kind, `row_record` union member, `hm_record_type`, `hm_unify_rows` declarations |
| `src/type_infer.c` | Row type construction, unification, pretty-printing, occurs check, spread inference, open-pattern inference |
| `tests/test_row_poly.c` | Unit tests for all row polymorphism features |
| `examples/row_poly_records.nano` | Example programs showing row-polymorphic idioms |
| `examples/agentos_ipc_schema.nano` | agentOS IPC schema example using row records |

### Limitations / Future Work

- Row variables are not yet surfaced in error messages with the `ρ` symbol
  (they show as `α<n>` currently).
- The parser does not yet support `{f:T | r}` syntax natively; the HM
  layer models it via explicit `hm_record_type()` calls.  A parser extension
  is planned.
- Recursive record types (µ-types) are not supported; the occurs check
  prevents them.
- No subtyping: narrowing must be done by explicit field selection, not
  implicit coercion.
