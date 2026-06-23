# SEMANTICS.md — Evaluation + Compilation Semantics

## 1) Environments and scoping
Each block defines a scope with:
- `inputs`: names + optional defaults
- `locals`: `let` bindings
- `results`: aliases from `call ... as NAME`
- `outputs`: `emit` fields

No shadowing (enforced by validation).

## 2) Compilation: AST → Execution Plan
The compiler emits ordered steps (preserve source order).

### Step types
- `call`: invoke a tool with args
- `let`: bind a derived value (optional but recommended for simplicity)
- `assert`: runtime assertion
- `emit`: publish a named value
- `if`: conditional execution for `when`

### Encoding values in the plan
- Literal JSON values appear directly (e.g. `3`, `"x"`, `{...}`)
- Variable references: `{ "$var": "name" }`
- Compound expressions: `{ "$expr": <exprIR> }`

Rule:
- identifier → `$var`
- literal → literal
- everything else → `$expr`

### when rule
`when cond -> a1, a2`
compiles to:
`{ "op":"if", "cond": <value>, "then":[...], "else":[] }`

## 3) Capability gating
Compilation should assume validation already enforced:
`call TOOL` permitted only if `uses TOOL` exists in the block.

## 4) Determinism
- Preserve step order and action order
- Preserve map key order from AST `entries[]` where possible
