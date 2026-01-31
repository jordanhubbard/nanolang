# BUILTINS.md — Optional Expression Builtins

OPL optionally supports pure functions in expressions via `call_expr`, like:
`len(x)`.

These are NOT tool calls. They must be deterministic and side-effect free.

## Suggested whitelist
- `len(x)`  (strings/lists/maps)
- `lower(s)`, `upper(s)` (strings)
- `has(map, key)` (maps)

If you implement none, reject callExpr with `E_FEATURE_NOT_IMPLEMENTED`.
