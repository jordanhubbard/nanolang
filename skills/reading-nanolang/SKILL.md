---
name: reading-nanolang
description: >-
  Read, understand, and write current NanoLang (.nano) source correctly. Use
  this whenever you inspect, review, or generate NanoLang, including src_nano/
  and NanoLang modules. Covers current calls, equal-precedence operators, local
  inference, optional else, project shadow policy, safety boundaries, and my
  first-person voice.
---

# Reading And Writing NanoLang

Read [`docs/CANONICAL_STYLE.md`](../../docs/CANONICAL_STYLE.md) before writing
`.nano` code. It is my implementation-backed best-practices guide and separates
syntax enforced by the compiler from repository policy. Read
[`docs/PERSONA.md`](../../docs/PERSONA.md) before producing user-facing text.

## Rules To Keep In View

- Write calls as `(function arg1 arg2)` and qualified calls as
  `(module.function arg1)`. Operators accept prefix and infix forms.
- All infix binary operators, including equality, comparison, and boolean
  operators, have equal precedence and associate left to right. Parenthesize
  intended grouping.
- Write immutable `let` bindings by default, `let mut` when reassignment is
  needed, and mutate with `set`.
- Local annotations are optional: `let value = expression` is implemented type
  inference. Function parameters and return types remain explicit. Annotate
  locals where generic, empty, foreign, union, or resource types need clarity.
- A bare `if condition { ... }` is valid. `else` is optional. `cond` requires a
  final `(else value)` clause.
- Prefer `module "path.nano" as alias` and qualified public names for new code.
  Legacy `import` and `from ... import ...` forms still parse.
- Direct extern calls require `unsafe { ... }` unless the module is unsafe.
  Keep unsafe and FFI boundaries narrow.
- Resource checking is partial, not a complete ownership proof. Explicitly
  annotate resource locals, consume each once, and audit every cleanup path.
- The compiler normally warns about missing shadows and has exemptions. Project
  policy still requires a useful shadow for every added or changed non-extern
  named function. A shadow is a test, not a proof.
- `#`, `//`, and `/* ... */` comments are accepted. Prefer `#` for ordinary
  comments and reserve `///` for documentation tooling.
- Use `snake_case` for values and functions and `UpperCamelCase` for named types
  and variants.

## Before Making A Claim

- Read the whole relevant function and its imports. Do not infer structure from
  a truncated search result.
- Check the current parser or typechecker before declaring syntax invalid.
- Compile the changed program with the repository's current compiler and run
  relevant shadows or tests.
- Say **proved** only for a checked formal theorem in its stated subset, **tested**
  only for behavior a named test exercised, and **assumed** for unchecked,
  platform, FFI, graphical, or policy-dependent claims.
- Keep example metadata consistent with the header fields documented in
  `docs/CANONICAL_STYLE.md`; do not invent successful outputs or dependencies.

All documentation, examples, diagnostics, and user-facing comments speak in my
first person: "I compile to C," not "NanoLang compiles to C." My tone is direct,
plain, and precise.
