# VALIDATION.md — Static Validation Rules

Validator input: AST (AST_IR.schema.json)  
Validator output:
- OK: `{ "ok": true }`
- Errors: `{ "ok": false, "errors": [ {code, msg, loc, path} ... ] }`

Errors must be deterministic in ordering:
1) sort by `loc.line`, `loc.col`
2) then by `code`
3) then by `path` (lexicographic)

## V1 — Structural Rules

### V1.1 Duplicate names in same scope
- Block names must be unique at program top-level per blockType.
- Within a block:
  - `let` names must be unique (no shadowing).
  - `call ... as NAME` names must be unique and must not conflict with `let` or params.
  - `emit NAME:` must be unique.

### V1.2 Declaration placement
- `uses`, `input`, `output`, `returns`, `doc` are only valid inside a block.
- `include` is allowed only at top-level (optional feature).

### V1.3 Uses gating (capabilities)
- For each `call REF ...`, REF must be present in a `uses REF` declaration in the nearest enclosing block.
- If not, error `E_CALL_NOT_ALLOWED`.

## V2 — Name resolution

### V2.1 Identifiers
Any identifier used in an expression must resolve to one of:
- an `input` param name
- an `output` param name (optional)
- a `let` name in scope
- a `call` result alias (`as NAME`)

Unresolved identifier => `E_UNRESOLVED_ID`.

## V3 — Type rules (light but strict)

### V3.1 Operator typing
- `and/or/not` require boolean operands.
- `==/!=` allow any JSON types but no implicit coercion.
- `< <= > >=` require numbers (default).
- `+ - * /` require numbers by default.
  - OPTIONAL: allow `+` to concatenate strings.

Type errors => `E_TYPE_MISMATCH`.

### V3.2 Map keys
Duplicate keys in `{ key: expr, key: expr }` => `E_DUPLICATE_KEY`.

### V3.3 Tool-call argument values
After evaluation, tool call args must be JSON-compatible:
string/number/bool/null/list/map.

## V4 — Control-flow rules
- `when/on` must have at least one action
- `assert` condition must be boolean
