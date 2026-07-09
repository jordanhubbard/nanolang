# SPEC.md — OPL (Operator Prompt Language)

## Purpose

OPL is a tiny, line-oriented DSL to express:
1) **Concise prompts** (declarative intent)
2) **Explicit tool calls** (side effects)
3) **Executable specifications** (agents/services/tasks) that compile into an **Execution Plan**

OPL is designed for:
- **Unambiguous parsing**
- **Deterministic compilation**
- **Minimal typing** (few keywords, lightweight punctuation)
- **Auditability** (capability gating via `uses`)

## Non-goals

- Being a natural language replacement
- Covering all computation (OPL is not a general-purpose language)
- Implicit “helpfulness” or inference. If it’s not explicit, it’s not assumed.

## Key Concepts

### Blocks
Top-level constructs with scoped declarations and statements.

```
agent NAME { ... }
service NAME { ... }
task NAME { ... }
schema NAME { ... }   # optional
```

### Declarations (inside blocks)
- `uses TOOLREF`              capability declaration
- `input name:type (= default)? (, ... )`
- `output name:type (= default)? (, ... )`
- `returns type`              optional
- `doc "string"`              optional

### Statements
- `let name = expr`
- `call tool.ref { k: expr, ... } as name?`
- `when expr -> action (, action)*`
- `on event(...) -> action (, action)*`   # optional
- `assert expr else "message"?`
- `emit name: expr`
- `include "file"`            # optional (for multi-file; may be ignored in flat examples)

### Expressions
Small JSON-like expressions plus boolean operators:
- literals: string, number, true/false, null
- identifiers
- lists: `[a, b]`
- maps: `{k: v}`
- member access: `x.y`
- operators: `not`, `and`, `or`, `==`, `!=`, `<`, `<=`, `>`, `>=`, `+`, `-`, `*`, `/`
- optional: pure builtins like `len(x)` (whitelist)

## Determinism Rules

1. Statement order matters for compilation to plans (preserve source order).
2. No implicit coercions: operations on mismatched types must be validation errors.
3. Maps preserve insertion order as written (AST must retain ordering).

## Capability Gating

A statement `call T` is legal only if the nearest enclosing block contains `uses T`.

## Compilation Output

Compilation yields an **Execution Plan** — a JSON structure describing tool calls and control flow.
No execution occurs in this compilation stage.

See: PLAN_IR.schema.json and SEMANTICS.md.
