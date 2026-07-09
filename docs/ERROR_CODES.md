# Compiler Error Codes

The nanolang typechecker tags every diagnostic with a stable `EXXX` code
in the `-- <code> <category>` banner so callers can match on the code
rather than the prose. The code stays stable across releases; the
message text may evolve.

## Quick reference

| Code | Category | Description |
|------|----------|-------------|
| **E001** | TYPE MISMATCH | An expression's type doesn't match the declared/expected type. |
| **E002** | PURITY VIOLATION | A `pure fn` calls a non-pure function, mutates state, or performs I/O. |
| **E003** | ARITY MISMATCH | A function/constructor was called with the wrong number of arguments. |
| **E004** | UNKNOWN FIELD | Field access uses a name that doesn't exist on the struct/union variant. |
| **E005** | UNKNOWN VARIANT | A `match` arm or constructor names a variant that doesn't exist on the union. |
| **E006** | UNDEFINED STRUCT | A struct type name is referenced but no `struct <Name>` is in scope. |
| **E007** | UNDEFINED UNION | A union type name is referenced but no `union <Name>` is in scope. |
| **E008** | SYNTAX ERROR | The typechecker received an AST shape it can't make sense of (usually a parser-side issue surfacing here). |
| **E009** | PRIVATE ACCESS | Code in module B references a `priv` (private) symbol from module A. |
| **E010** | MISSING RETURN | A non-void function has a control-flow path that does not return a value. |

## Example diagnostic

```
-- E001 TYPE MISMATCH ------------- examples/demo.nano
    3 |     let x: int = "hello"
                          ^^^^^^^
Expected int, found string.
Hint: convert with int_of_string, or change the declared type.
```

## How to grep

These codes are stable identifiers, so you can grep build output for
them in CI:

```bash
make examples 2>&1 | grep -oE '^-- E[0-9]+' | sort | uniq -c | sort -rn
```

If a new code is added, append it to the table here and to the
emit_context_error call sites in `src/typechecker.c`. Codes are
assigned sequentially in approximate order of historical frequency
(E001 = most common); never renumber an existing code.

## Adding a new code

1. Pick the next free `EXXX` number.
2. Use it in your `emit_context_error("EXXX SHORT TITLE", ...)` call.
3. Add a row to the table above with the category and a one-sentence description.
4. If the code documents a class of error rather than a single check
   site, link the relevant `src/typechecker.c` function from the table.

## Related

- `src/typechecker.c::emit_context_error` — the formatter that prints
  the banner and renders the source-line context.
- `bd show nl-imn.1.2` — bead tracking the introduction of E-codes.
