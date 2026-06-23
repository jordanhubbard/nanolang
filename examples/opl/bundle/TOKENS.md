# TOKENS.md — Lexical Rules

OPL is line-oriented. Newlines can terminate statements.

## Whitespace
- Spaces and tabs separate tokens.
- Newlines are significant as statement terminators (unless inside `{ ... }`, `[ ... ]`, or `( ... )`).
- Optional `;` may terminate a statement before a newline.

## Comments
- `#` begins a comment to end-of-line (unless inside a string).

## Identifiers
IDENT matches:
- First char: `[A-Za-z_]`
- Next chars: `[A-Za-z0-9_./:-]*`

Examples:
- `web.search`
- `http.get`
- `agent_1`

Note: literal paths like `/login` must be written as strings: `"/login"`.

## String literals
Double-quoted strings with escapes:
- `\n`, `\t`, `\r`, `\"`, `\\`
- Optional: `\uXXXX` (Unicode escape)

## Numbers
- Integers: `0`, `123`, `-7`
- Floats: `1.0`, `-0.25`, `2e10`, `-3.5E-2`

## Keywords (reserved)
agent, service, task, schema,
uses, input, output, returns, doc,
let, call, as, when, on, assert, else, emit, include,
true, false, null,
and, or, not

## Operators and punctuation
`=` `:` `,` `->` `{` `}` `[` `]` `(` `)` `.` `==` `!=` `<` `<=` `>` `>=` `+` `-` `*` `/`

## Newline handling
Newlines terminate a statement unless inside `{}` `[]` `()`.
Backslash line-continuations are OPTIONAL; implementations may omit.
