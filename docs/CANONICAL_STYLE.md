# How I Prefer To Be Written

This guide describes the source my current parser and typechecker accept. It
also states project policy. Those are different things.

- **Accepted** means my current implementation parses and typechecks the form.
- **Required** means the implementation rejects the alternative.
- **Policy** means contributors should follow the rule even where I do not yet
  enforce it.

My implementation is the authority when this guide and old examples disagree.
The relevant code lives in `src/lexer.c`, `src/parser.c`, `src/typechecker.c`,
and `src/resource_tracking.c`. Tests show observed behavior. Neither a parser
branch nor an example is a proof of the whole language.

## Calls And Expressions

Write calls in parenthesized prefix form. Arguments are separated by spaces,
not commas.

```nano
(println "ready")
(distance x y)
(canvas.clear color)
(clock_now)
```

I parse `(name)` as a zero-argument call. Parentheses around a non-identifier
expression are grouping. Commas make tuples:

```nano
let point = (10, 20)
let answer = (compute)
let grouped = (a + b)
```

Do not write `f(x, y)` or `object.method()`. They are not my call syntax. Dot
syntax names struct fields and qualified module members; the call still wraps
the qualified name: `(math.clamp x low high)`.

Operators are the deliberate exception to call-only syntax. I accept prefix
and infix forms for `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `<`, `<=`, `>`, `>=`,
`and`, and `or`. I accept unary `not` and unary `-`.

```nano
let prefix = (+ a b)
let infix = a + b
let negative = -value
let disabled = not enabled
```

All infix binary operators, including equality and boolean operators, have
equal precedence and associate left to right:

```nano
let first = 2 + 3 * 4       # (2 + 3) * 4
let second = 2 + (3 * 4)    # 2 + (3 * 4)
let flag = a == b and ready # (a == b) and ready
```

**Policy:** prefer prefix operators in dense or mixed expressions. If infix is
clearer, parenthesize every intended grouping. I have no hidden precedence
table to rescue an ambiguous-looking line.

## Bindings And Types

Bindings are immutable unless marked `mut`. Mutation uses `set`.

```nano
let name = "Ada"
let count: int = 3
let mut frame = 0
set frame (+ frame 1)
```

Local type annotations are optional. For `let x = expression`, my typechecker
infers the type from the initializer and records it for later compiler phases.
This is current behavior, not "limited future support."

**Policy:** use local inference when the initializer makes the type plain. Add
an annotation for empty collections, generic or union-heavy values, FFI
handles, resource values, and anywhere the annotation documents a boundary.
Some collection constructors require an annotation because their element types
cannot be recovered from an empty value.

Function parameters and return types remain explicit and required:

```nano
fn area(width: int, height: int) -> int {
    return (* width height)
}
```

Use the implemented type spellings rather than inventing host-language ones.
Common forms include `int`, `float`, `bool`, `string`, `void`, `array<T>`,
tuples such as `(int, string)`, named `struct`, `enum`, `union`, generic unions
such as `Result<int, string>`, function types, and opaque FFI types. Construct
named records with `Type { field: value }` and read fields with `value.field`.
Named types and variants start with an uppercase letter because the parser uses
that convention when distinguishing constructors from ordinary identifiers.

## Control Flow

An `if` may omit `else`:

```nano
if needs_redraw {
    (draw scene)
}
```

I also accept `else if` and `else`. Do not add `else {}` merely to satisfy an
old document. Use an `else` when both outcomes matter.

`cond` is an expression with a required final `(else value)` clause:

```nano
let sign = (cond
    ((< n 0) -1)
    ((> n 0) 1)
    (else 0)
)
```

Use `if` for ordinary control flow and `cond` when selecting a value from
several cases. Use `match` for unions and other data whose variants carry the
decision. Current `for` syntax iterates a range expression:

```nano
for i in (range 0 count) {
    (visit i)
}
```

`while`, `break`, and `continue` are also accepted.

## Imports And Modules

The current module import form is `module`:

```nano
module "modules/std/json/json.nano" as json

fn decode(payload: string) -> Json {
    return (json.parse payload)
}
```

A bare module identifier resolves through `modules/<name>/<name>.nano`, but an
explicit path is easier to audit. The parser still accepts legacy `import`,
`from ... import ...`, wildcard imports, and `pub use`. Do not choose legacy
syntax for new code merely because it still parses.

**Policy:** import a module under a short, specific alias and qualify its public
API. Use selective imports only when they materially improve a small file.
Avoid wildcard imports. Keep imports at the top, keep private helpers private,
and mark only the intended module surface `pub`. A module is a boundary: expose
domain types and operations, not incidental storage or raw foreign calls.

## Unsafe Code And FFI

Declare foreign functions with `extern fn`. A direct call to one must occur in
an `unsafe { ... }` block unless the enclosing module is unsafe.

```nano
extern fn c_read(fd: int, buffer: string, count: int) -> int

fn read_once(fd: int, buffer: string) -> int {
    unsafe {
        return (c_read fd buffer (str_length buffer))
    }
}
```

An unsafe import marks the imported module as unsafe:

```nano
unsafe module "modules/sdl/sdl.nano" as sdl
```

**Policy:** keep unsafe regions small. Put FFI declarations and representation
conversions in a boundary module, validate values there, and export a typed safe
wrapper where one can honestly be provided. Use an unsafe module only when the
whole module is a foreign boundary. `unsafe` records responsibility; it does
not make a pointer valid or a C library well behaved.

## Resources

`resource struct` marks an affine resource type:

```nano
resource struct FileHandle {
    fd: int
}

fn close_file(file: FileHandle) -> void {
    unsafe { (c_close file.fd) }
}
```

The current typechecker marks explicitly typed resource bindings and treats a
direct resource identifier passed by value as consumed. It contains checks for
use-after-consume and repeated consumption. The implementation is not a full
ownership proof: leak checking is not wired into normal typechecking, inferred
resource bindings do not carry all the same metadata, branch-sensitive state
is limited, and some resource diagnostics do not currently propagate into the
typecheck result.

**Policy:** annotate resource locals explicitly, give ownership to exactly one
scope, pass the resource once to its cleanup function, and never use it
afterward. Review every return and early exit for cleanup. Treat successful
compilation as a check, not proof of resource safety.

## Errors

Use `Result<T, E>` or another explicit union for failures callers can handle.
Match the variants where context exists to recover or report. The postfix `?`
operator is accepted for compatible result propagation, but an explicit
`match` is better when adding context or cleanup.

Use `assert` in shadows for invariants and expected behavior. Do not use an
assertion as routine input validation. At process boundaries, print a concise
diagnostic and return a nonzero `int` from `main`. At module boundaries, return
typed error information instead of magic integers where practical. Preserve
foreign error codes or messages before performing more FFI calls.

## Shadows: Enforcement And Policy

A shadow is an executable test attached by name:

```nano
fn double(value: int) -> int {
    return (* value 2)
}

shadow double {
    assert (== (double 3) 6)
}
```

I do **not** universally reject a function without a shadow. In normal CPU
typechecking I warn. I exempt `extern` functions, `main`,
generated lambdas, GPU targets, and functions whose bodies call extern
functions. It rejects a shadow attached directly to an `extern` function.

Project policy is stricter than compiler enforcement: every added or changed
non-extern named function gets a useful shadow in the same change, including
`main` and wrappers around FFI when they can be tested. The repository's shadow
check script enforces this policy textually for added functions in new files
and functions added in a diff. That script is a policy gate, not the language
grammar, and it does not establish test quality.

Write shadows around observable contracts. Include boundaries and failure
variants. A shadow shows behavior for the cases it executes. It does not prove
the function for all inputs.

## Comments And Names

I accept three comment forms:

```nano
# ordinary line comment
// line comment; /// is available to documentation tooling
/* block comment */
```

Prefer `#` for ordinary source commentary. Use `///` only for API documentation
consumed by documentation tools. Use block comments sparingly. Explain a
constraint, ownership rule, foreign assumption, or non-obvious reason. Do not
narrate syntax.

Use `snake_case` for functions, locals, fields, and module aliases. Use
`UpperCamelCase` for structs, unions, enums, and their variants. Use precise
nouns for values and verbs for operations. Keep abbreviations established by
the domain or foreign API. Prefix private helpers by domain when a generic name
would be hard to search; do not encode types into names.

## Example Metadata

Repository examples begin with machine-readable `# Key: value` lines. Keep the
catalog vocabulary and order used by neighboring current examples:

```nano
# Example: Checked Parser
# Purpose: Parse input and report a typed error
# Features: unions, match, shadow tests
# Difficulty: Intermediate
# Category: language
# Prerequisites: nl_union_types
# Track: learn
# Build: local
# Dependencies: none
# Tags: parsing, result, shadow-tested
# Expected Output: parsed 3 records
```

`Example`, `Purpose`, `Features`, `Difficulty`, `Category`, `Prerequisites`, and
`Expected Output` are the common core. Current catalog entries may also use
`Track`, `Build`, `Dependencies`, and `Tags`. Use `none` rather than leaving a
known-empty field blank. Metadata is catalog data in comments; my parser
ignores it. Do not claim a backend, dependency, output, or shadow status
that has not been checked.

## Graphical Programs

A graphical example has a lifecycle, not merely a loop:

1. Import the foreign modules and initialize each subsystem.
2. Check every required window, renderer, context, font, texture, or audio
   handle before entering the loop.
3. Poll all pending events, then update state, then render and present once.
4. Keep per-frame allocation and foreign resource creation out of the loop.
5. Leave the loop through one cleanup path and destroy resources in reverse
   construction order.

Use `# Expected Output: graphical` and list external libraries in
`Dependencies`. Put `graphical` and `external-deps` in `Build` when applicable.
A graphical shadow should test pure state transitions, geometry, parsing, or
other deterministic helpers. Do not pretend that opening a window is a stable
unit test. The interactive loop is tested by running it on a supported system;
the helper mathematics can often be proved or tested more strongly.

## Claims: Proved, Tested, Assumed

I use these words narrowly:

- **Proved:** a stated property is covered by a checked formal theorem, within
  that theorem's model and subset.
- **Tested:** a named command, shadow, or test exercised stated cases and
  passed on a stated backend or environment.
- **Assumed:** the claim depends on an unchecked invariant, foreign library,
  platform behavior, manual review, or intended implementation behavior.

Do not write "proved" when a shadow passed. Do not write "tested" when code
only compiled. Do not extend a proof about my verified subset to FFI, graphics,
resource cleanup, or another backend without a theorem that covers it. State
the boundary. Honesty is more useful than confidence.

## Working Rule

Write the smallest explicit program that states its boundaries:

- Prefix calls; grouped operators.
- Inferred obvious locals; explicit public and difficult types.
- Optional `else` used only when there is another outcome.
- Qualified current-form module imports.
- Narrow unsafe and FFI boundaries.
- Explicit ownership and cleanup for resources.
- Typed recoverable errors.
- Shadows required by project policy, described as tests rather than proofs.
- Metadata and claims that report what was actually checked.

That is my current style. When my implementation changes, update this guide
from the parser and typechecker before repeating an older rule.
