---
name: reading-nanolang
description: >-
  Read, understand, and write NanoLang (.nano) source correctly. Use this
  whenever you open .nano files, review NanoLang code, or generate NanoLang —
  including the compiler's own source in src_nano/ and modules written in
  NanoLang. Covers the prefix/infix call syntax, mandatory shadow tests,
  explicit typing, required else branches, and NanoLang's first-person voice.
---

# Reading and writing NanoLang

NanoLang is a small, explicit language that compiles to C (and to NanoISA
bytecode). It is deliberately unambiguous: one canonical form per construct,
mandatory tests, explicit types. Do not read it as if it were Python, C, or
Lisp — the rules below are what actually hold.

## The syntax that trips people up

**Calls are prefix, in parentheses.** `(f x y)` calls `f` with `x` and `y`.
This includes operators and built-ins: `(println "hi")`, `(+ a b)`,
`(str_length s)`. When you grep NanoLang with tools that assume C/Python, short
identifiers and prefix calls can *look* garbled or obfuscated — they are not.
Read the file, don't guess from a mangled grep.

**Operators work in prefix or infix, with EQUAL precedence, left-to-right.**
`2 + 3 * 4` evaluates left-to-right as `(2 + 3) * 4 = 20`, NOT 14. There is no
precedence table. Write `2 + (3 * 4)` when you mean 14. Prefer parentheses.

**Every function MUST have a shadow test.** A `fn` without a matching `shadow`
block will not compile. Shadow tests are inline assertions that run when the
binary executes:

```nano
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 3) 6)
    assert (== (double 0) 0)
}
```

The only exception is `extern fn` (C code NanoLang cannot see). When you add or
edit a function, add or update its shadow test in the same change.

**Types are explicit and immutable by default.**

```nano
let x: int = 42                 # immutable
let mut counter: int = 0        # mutable — `mut` is required to reassign
set counter (+ counter 1)       # `set` only works on `mut` bindings
```

Function parameters and return types are always annotated:
`fn add(a: int, b: int) -> int`.

**`if` requires an `else` branch.** There is no bare `if`. Every `if` has a
matching `else { ... }` (which may be empty: `else {}`).

**Structs use dot access and literal construction:**

```nano
struct Point { x: int, y: int }
let p: Point = Point { x: 1, y: 2 }
let sum: int = (+ p.x p.y)
```

## Imports and modules

```nano
from "modules/std/fs.nano" import read, basename          # selective
from "modules/std/env.nano" import get as env_get          # with alias
```

`extern fn name(...) -> T` declares a C function; calls to it must sit inside an
`unsafe { ... }` block.

## The persona — write in NanoLang's voice

All user-facing text (docs, examples, error messages, comments meant to be read)
speaks as NanoLang, in the first person: "I compile to C", not "NanoLang
compiles to C". The tone is direct, plain, and unhurried — no marketing
language, no superlatives. Distinguish what is proved from what is tested. Show
code over prose. Read `docs/PERSONA.md` in full before producing user-facing
text.

## Before you claim something about NanoLang code

- Open the file and read it. Do not infer structure from a truncated or mangled
  search result.
- If a function has no shadow test, that is a bug (or it is `extern`).
- If you're editing, keep the shadow test true and add cases for new behavior.
- Verify with the compiler: `bin/nanoc_c <file>.nano -o /tmp/out` compiles and
  runs the shadow tests. `make shadow-check` verifies shadow tests on changed
  files.
