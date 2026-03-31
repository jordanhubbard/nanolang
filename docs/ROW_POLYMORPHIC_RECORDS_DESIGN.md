# Row-Polymorphic Record Types for NanoLang

**Status:** Design proposal (2026-03-31)  
**Author:** Rocky / Natasha (Natasha idea: wq-NAT-idea-1774933699352-NANO-RECORDS)  
**Motivation:** Typed agent message schemas for agentOS — agents need to exchange records
with different capability sets while the type system enforces required fields without
rejecting extended records.

---

## Problem

NanoLang currently has nominal struct types. A function that accepts `Agent { name: String }`
only accepts exact `Agent` structs. This breaks in agentOS where:

- A `CapabilityMessage` might extend a `BaseMessage` with extra fields
- Type-safe IPC requires both sender and receiver to agree on required fields
- But senders at different privilege levels add different optional fields

Structural subtyping solves this: a function that requires `{ name: String }` should
work on any record that *has at least* a `name` field.

---

## Design

### Row Variables

A **row variable** `r` stands for "zero or more additional fields." An open record type
`{ name: String, age: Int | r }` is read as "a record with at least `name: String` and
`age: Int`, plus whatever other fields `r` represents."

A **closed record type** `{ name: String, age: Int }` has no row variable — exact fields only.

```nanolang
// Open record type — accepts any record with at least a "name" field
fn greet({ name: String | r }) -> String {
    "Hello, " + name
}

// Works on exact records
let person = { name: "Alice", age: 30 };
greet(person)  // ok

// Also works on extended records
let agent = { name: "Rocky", role: "commander", host: "do-host1" };
greet(agent)  // ok — extra fields satisfy the open row
```

### Syntax

```
RecordType   ::= '{' FieldList '}'
               | '{' FieldList '|' RowVar '}'
FieldList    ::= FieldDecl (',' FieldDecl)*
FieldDecl    ::= IDENT ':' Type
RowVar       ::= IDENT   -- lowercase, like 'r', 'rest', 'extras'

RecordExpr   ::= '{' FieldInits '}'
               | '{' '...' Expr ',' FieldInits '}'   -- record spread
FieldInits   ::= FieldInit (',' FieldInit)*
FieldInit    ::= IDENT ':' Expr

RecordPattern ::= '{' PatternFields '}'
                | '{' PatternFields ',' '..' '}'    -- open pattern
PatternFields ::= PatternField (',' PatternField)*
PatternField  ::= IDENT ':' Pattern
                | IDENT                              -- shorthand {name} = {name: name}
```

### Subtyping Rules

1. `{ f1: T1, f2: T2 } <: { f1: T1 }` — closed record is subtype of closed with fewer fields
2. `{ f1: T1, f2: T2 } <: { f1: T1 | r }` — closed record satisfies open row
3. `{ f1: T1 | r } <: { f1: T1 }` is *not* generally valid (open is not subtype of closed)

### Record Spread

```nanolang
let base = { x: 1, y: 2 };
let extended = { ...base, z: 3 };  // { x: Int, y: Int, z: Int }
```

Spread produces a closed record; the type is the union of all fields.

---

## Implementation Plan

### Phase 1 — Type Representation (`src/typechecker.c`)

Extend `NanoType` with row variable support:

```c
// Existing:
typedef struct { ... } NanoType;

// Add to the struct union:
struct {
    FieldList *fields;   // linked list of (name, type) pairs
    int row_var_id;      // -1 = closed, >=0 = open row variable
} record_type;
```

Row variables participate in the union-find structure already used for
HM-style type inference. A row variable unifies with another row variable
(merging their field sets) or with a concrete empty extension (closing the row).

### Phase 2 — Row Unification

Extend `unify()` in `typechecker.c`:

```c
// Unify two record types
static int unify_records(NanoType *a, NanoType *b) {
    // Each required field in a must exist in b with compatible type
    for each field (name, type) in a->fields:
        if (b has field name):
            if (!unify(type, b->field_type(name))) return 0;
        elif (b->row_var_id >= 0):
            // Add field to b's row variable extension
            extend_row(b->row_var_id, name, type);
        else:
            return 0;  // closed record missing required field
    
    // If a has a row variable, propagate remaining b fields into it
    if (a->row_var_id >= 0):
        for each field in b not already matched:
            extend_row(a->row_var_id, field.name, field.type);
    
    return 1;
}
```

### Phase 3 — Parser (`src/parser.c`)

1. Add `TYPE_RECORD` AST node with optional row variable
2. Add record expression `{ field: expr, ... }` (may already exist as struct literal)
3. Add record spread `{ ...expr, field: val }`
4. Add open record pattern `{ field: pattern, .. }`

### Phase 4 — Pattern Matching (`src/typechecker.c` + exhaustiveness)

Open record patterns `{ name: String, .. }` match any record with at least a `name` field.
Exhaustiveness checker: an open pattern is always exhaustive; a closed pattern must cover
all record constructors.

### Phase 5 — Code Generation

Records are structs in the C backend. Open record types compile to:
- A `void*` fat pointer + field accessor functions, OR
- A tagged struct with dynamic field map (simpler, slightly slower)

Recommended: **tagged struct** for correctness in v1, optimize later with static dispatch
once row inference is proven.

---

## Test Cases

```nanolang
// File: tests/test_records.nano

// Basic record creation
let p = { name: "Alice", age: 30 };
assert(p.name == "Alice");

// Open-row function
fn getName({ name: String | r }) -> String { name }
assert(getName({ name: "Bob", extra: 99 }) == "Bob");

// Record spread
let base = { x: 1, y: 2 };
let ext = { ...base, z: 3 };
assert(ext.x == 1 && ext.z == 3);

// Open pattern match
match p {
    { name: n, .. } => print("Name: " + n),
}

// agentOS message schema example
type BaseMsg = { seq: Int, from: String | r };
type CapMsg  = { seq: Int, from: String, cap: String };
fn routeMsg(m: BaseMsg) -> Unit { print(m.from) }
routeMsg({ seq: 1, from: "rocky", cap: "read" });  // CapMsg satisfies BaseMsg
```

---

## Relation to agentOS

Agent PD messages in agentOS use C structs with a fixed header + variable payload.
Row-polymorphic records mirror this exactly:

```nanolang
type AgentMsg = {
    seq:  Int,
    from: String,
    // 'rest' row var — each PD adds its own fields
    | rest
};
```

Functions that route or log messages work on `AgentMsg`; functions that grant capabilities
work on `{ seq: Int, from: String, cap: String }` (a subtype). The type system enforces
"you must provide seq and from" without forcing all code to know about capabilities.

---

## Dependencies

- `a236ab17` — Pattern matching (exhaustiveness, guards, wildcards) ✅
- HM-style type inference — partial (typechecker.c has constraint-based checking but not
  full algorithm W). Row unification can reuse the existing union-find; full HM is
  not strictly required for row polymorphism if we restrict row variables to function
  parameters (monomorphic row instantiation).

---

## Effort Estimate

| Phase | Estimated effort |
|-------|-----------------|
| Type representation | 0.5d |
| Row unification | 1d |
| Parser | 0.5d |
| Pattern matching | 0.5d |
| Code generation (tagged struct) | 1d |
| Tests | 0.5d |
| **Total** | **~4d** |
