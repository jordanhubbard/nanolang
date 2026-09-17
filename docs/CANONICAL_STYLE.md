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

## Function Values And Captures

My NanoVirt backend instantiates anonymous functions at their lexical expression,
including when the C parser hoists their declarations internally. Named nested
functions and anonymous functions retain captures through intermediate closures.
I test returned closure chains and callbacks nested across native waits.

```nano
fn make_offset(n: int) -> fn() -> fn() -> int {
    return fn() -> fn() -> int {
        return fn() -> int { return (+ n 2) }
    }
}

shadow make_offset {
    let middle: fn() -> fn() -> int = (make_offset 40)
    let inner: fn() -> int = (middle)
    assert (== (inner) 42)
}
```

This is tested bytecode behavior, not a claim that the C-native or tree-walking
backends implement the same anonymous-capture semantics.

My self-hosted native compiler also supports computed calls such as
`((choose) value)` when `choose` returns a function. I test nested calls,
nested function-parameter signatures, wrong argument types and counts, and
callee-before-arguments evaluation exactly once from left to right. These
tests use named function values; they do not establish captured-closure parity.
My self-hosted NanoISA subset still rejects computed calls explicitly.

My C-seed checker derives a mapped array's element type from the transform's
declared result, including direct indexing and local-bound results. I check
two map arguments, a unary value-producing transform and matching scalar input
types. I test named, variable and returned transforms. Complete nominal/nested
callback metadata and cross-backend result storage remain roadmap work; these
frontend checks do not establish complete map runtime support.
My interpreter now allocates scalar map results from the transform's declared
return type. I test int, float, bool and string results with static/dynamic
input arrays, empty/nonempty inputs and unchanged input values. My C-native and
VM lowering now use the same declared scalar result type when allocating map
output. I test all 16 scalar input/result pairs on both paths, including named,
variable and returned transforms and append-after-empty. This does not establish
nominal/nested callback metadata or aggregate map support.
My self-hosted native emitter now passes the same 16 scalar input/result
pairings with named, variable and returned transforms, empty results and
append-after-empty. It reads and writes the two declared representations
independently. A separate trace checks that source and transform expressions
run once in that order, then the callback once per element. Complete
self-hosted signature validation and aggregate map parity remain work.
My self-hosted checker now rejects wrong map arity, non-array sources,
non-function, non-unary and void transforms, and known input-element types
that disagree with the transform parameter. It infers the mapped array from
the transform result and checks direct map result annotations before C
emission. I test eight rejected programs with prior-output preservation and
an unannotated float-result program through shadows and native execution.
My general checker now rejects mismatched known scalar arrays through aliases,
function arguments, returns and reassignment. I test all 12 mismatched scalar
pairs on those four paths before C emission, preserving prior output, and
run typed empty-array/append/identity cases for each scalar type. Both compiler
stages rebuild without the former `array<int>` wildcard. Unknown-type
compatibility and recursive/nominal element comparisons remain incomplete;
this is not a claim of full type soundness.

My self-hosted parser now keeps nested literal element spans separate. My
checker retains and compares nested array descriptions, and my native emitter
keeps the remaining array depth through indexing and the storage type through
empty nested literals. I test 24 nested mismatches at their intended diagnostic
boundaries, three-level integer values, uneven/empty rows and append to empty
inner float arrays. Nominal/generic execution and unknown compatibility remain
open; these cases do not establish complete array soundness or backend parity.

## String Literals

Ordinary quoted strings keep braces literally. Use the `f` prefix to interpolate:

```nano
let count: int = 7
let literal: string = "{count}"
let expanded: string = f"count {count}"
```

I lower f-strings in the shared lexer into concatenation and `to_string` calls.
My native emitter does not reinterpret ordinary strings as a second interpolation
syntax. I test long literals and more than 64 interpolation parts through native
and VM shadows and execution.

## String Search

`str_index_of` finds the first occurrence; `str_last_index_of` finds the last,
including overlapping matches. I return a byte offset or `-1` when no match
exists. An empty needle matches at zero for first-search and at the byte length
for last-search:

```nano
assert (== (str_index_of "ababa" "aba") 0)
assert (== (str_last_index_of "ababa" "aba") 2)
assert (== (str_last_index_of "abc" "") 3)
assert (== (str_index_of "éx" "x") 2)
```

My C-seed, self-hosted native emitter and VM builtin bridge share these search
helpers. They consume NUL-terminated strings; this is not Unicode-position
search or a length-aware search through embedded NUL bytes.

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

Block-local names stop being visible at the block's closing brace. My C
frontend retains their type metadata for emission without extending that
visibility. I test scalar and array shadowing through interpreted shadows and
native execution, including same-line blocks, loop exits and returned local
strings. These checks do not establish stored-closure lifetime safety or
complete scope identity for generated nodes.

### Match Values In 5.0

I use `return` to leave the enclosing function, including inside a match arm.
An expression arm yields that expression. A block arm yields its final
expression after executing its preceding statements:

```nano
let doubled = (match option {
    Some(item) => {
        if (< item.value 0) { return -1 }
        let value: int = (* item.value 2)
        value
    }
    None(empty) => { 0 }
})
return (+ doubled 100)
```

Here a negative value returns `-1` from the function; other paths continue
after the match. I do not use `return value` to yield an arm-local value.
My backend acceptance work, including control-flow and ownership boundaries,
is tracked in `docs/ROADMAP.md`.

### Effect Handler Parameters

I bind handler parameters from the declared operation signature, retaining
record names, nested array element types and scalar function signatures.
An unrelated outer variable with the same name does not supply the handler's
type. I test valid field access, array reads and callback calls, and reject
incompatible uses inside handler bodies.

This frontend checking does not establish complete perform argument/result
compatibility. Native and VM handler dispatch remain release work.

My 5.0 handler control-flow contract matches match arms: `return` exits the
function containing the handler, even when another function performs the
operation. The arm's final expression supplies the operation result and resumes
`perform`. My interpreter now preserves the lexical call destination through
ordinary helper calls and tested scalar expressions, including nested handler
unwinding and string returns. Aggregate constructors and match guards stop
before later elements or arms execute. I test partial string/record-array
cleanup without freeing the caller's original records. Interpreter `map`,
`filter` and `reduce` stop iteration when a callback propagates a handler return;
an ordinary handler value resumes the callback. Static and dynamic arrays are
tested. One hundred nested synchronous async calls also propagate the handler
return and release all 200 private coroutine handles. This is run-to-completion
interpreter behavior, not resumable async or an owned-task API. Full expression
coverage, foreign callback boundaries and native/VM lowering remain roadmap
work; these tests do not establish backend equivalence.

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

My C-seed and self-hosted parsers decode `\n`, `\t`, `\r`, `\\`, `\"`, and `\'` in quoted
import paths once, preserving unknown escapes as written. I reject `\0`:
a filesystem path cannot contain a NUL byte. These are path bytes, not shell
syntax. My self-hosted merger uses the same path decoder as its parser and
ignores quotes after the path's closing quote. Its line-oriented import
discovery is not a replacement for full syntax-aware module loading.

**Policy:** import a module under a short, specific alias and qualify its public
API. Use selective imports only when they materially improve a small file.
Avoid wildcard imports. Keep imports at the top, keep private helpers private,
and mark only the intended module surface `pub`. A module is a boundary: expose
domain types and operations, not incidental storage or raw foreign calls.

`std/fs.canonical` resolves an existing path to its physical absolute path,
including symlinks, and returns an empty string on failure. `std/fs.normalize`
remains lexical; it does not establish file identity. My self-hosted merger
uses physical identity before deduplication, relative dependency lookup and
function binding. This is path identity, not hard-link identity. My self-hosted
input/output alias guard separately compares device and inode identities.

My C-seed and bytecode paths now keep distinct pure functions with the same
short name across qualified imports. I test a root wrapper, a transitive
wrapper and their private helpers with inferred and declared module names.
I also scope import aliases by their declaring module in the C environment:
two wrappers may each call `lib.answer` without sharing the first wrapper's
binding. A dependency's alias is not visible at the root, and a root alias is
not inherited by unrelated modules. I test inferred and declared owners across
all three drivers. This uses module names, not canonical path identity.
My Stage2 driver now preserves pure function identity across that fixture too.
It retains import declarations, binds function references by their owning file,
and assigns imported function definitions private generated names. Selective
aliases do not rewrite strings, local bindings or field labels. I test reused
aliases in separate importers, function values and calls inside aggregates.
Local variables take precedence without leaking beyond their lexical scope.
I reject duplicate non-extern definitions within one file and unbound qualified
calls before native or C-source publication, preserving prior output. My JSON
codes are `M0001` and `M0002`; merged offsets are not original-file provenance.

This binding state belongs to one sequential compiler invocation; it is not
reentrant. Private visibility, conflicting aliases within one importer,
selective type aliases, colliding nominal module identities and foreign-name
isolation remain open. Same-basename fixtures use distinct declared module
identities; I still reject ambiguous introspection names. Qualification is not
yet a universal isolation guarantee across my backends.

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

extern fn consume_handle(owned: FileHandle) -> void

fn close_file(file: FileHandle) -> void {
    unsafe { (consume_handle file) }
}
```

The foreign declaration above states a consuming boundary; it is not an
implementation of cleanup. Reading `file.fd` alone does not consume `file`.
My C-seed checker tracks parameter and local obligations in a separate
function pass. I check moves, observations, scope exits, branch joins and loop
edges, and reject resource-bearing collections. My self-hosted checker now
passes the same bounded flow matrix, including short-circuit consumption and
hidden outer obligations. Its classifier propagates ownership through named
record fields and union payloads; generic substitution and module identity
remain incomplete. Resource match payloads, captures,
and borrows still need lowering; some forms are
rejected pending that work. Neither checker establishes the complete ownership
contract. Passing the shared flow gate is not full conformance, and successful
compilation does not establish runtime cleanup.

I support complete record patterns in local blocks:

```nano
fn handle_number(file: FileHandle) -> int {
    let FileHandle { fd } = file
    return fd
}
shadow handle_number {
    assert (== (handle_number FileHandle { fd: 42 }) 42)
}
```

I evaluate the initializer once, require the named record type and every field
exactly once, and move the source. Resource-bearing fields become new ownership
obligations; ordinary fields remain GC values. I reject omitted, duplicate and
unknown fields, `..`, ordinary partial resource-field moves and use of the
source afterward. I test native and VM shadows and execution, including unsafe
block scopes, nested resources, empty records and ordinary array fields.
This example dismantles a simulated handle; it does not close an OS file.

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

Execution is currently backend-dependent. My C seed checks dependency and root
shadows, then executes them in a separate interpreter child with a ten-second
parent deadline; foreign-call syntax does not exempt an explicit shadow.
My `nano_virt` CLI runs dependency and root-file shadows in a separate
verified NanoVM test module before publishing output, with a 10-second parent
deadline. Production bytecode omits that test harness. This child process is
not a security sandbox: tests can have side effects. My self-hosted native
driver also runs dependency and root shadows in a separate test executable with a ten-second
parent deadline before publishing native output. Test stdout is redirected to
stderr. Source-only `--target c` checks types but does not execute shadows or
invoke a native compiler. My creator chose dependency shadows by default for
5.0. My C-seed, bytecode and self-hosted native drivers implement that choice.

All three drivers include imports by default; `--test-imports` selects that
behavior explicitly, and `--root-shadows-only` opts out:

```sh
bin/nano_virt app.nano --test-imports --emit-nvm -o app.nvm
```

I check the selected dependency shadows before execution, then run them in
source-import traversal order: dependencies before importers, root last, each
loaded source file once. Within a file I preserve shadow source order. Existing
file paths are canonicalized before C-frontend cache and graph registration;
dot segments and symlinks do not select the same file twice. Relative imports
inside a symlinked module resolve from the target directory. This does not
establish colliding nominal type identity. My self-hosted merger uses the same
physical-path rule; it is still line-oriented import discovery, not a complete
syntax-aware loader.

All selected shadows share one test process and its ten-second parent deadline.
Foreign calls, printing and other side effects still happen with host authority;
the test child is not a sandbox. A failed type check, assertion, runtime trap or
deadline prevents publication. I do not put shadow entry functions in production
bytecode. Source-only C emission still does not execute tests. My native shadow
supervisor also requires its entry point to return successfully: calling
`exit(0)` or replacing the child with another executable is not test completion.
Set `NANO_SHADOW_TRACE=1` to print each self-hosted native shadow target before
it runs. Generated target names identify merged ownership, not original-file
diagnostic provenance.

Before self-hosted compilation writes output or diagnostics, I compare both
destinations with the root and loaded dependency files by device and inode.
I reject aliases through relative paths, symlinks and hard links, and stop if
file identity cannot be checked. This preflight covers stable filesystem
entries; it does not prevent concurrent path replacement. I also compare
artifact and diagnostic destinations by inode when both exist. When neither
exists, I exclusively create an empty directory at the requested output path,
look up the diagnostic spelling, compare identities, and remove the probe
before continuing. The filesystem decides case and Unicode name equivalence;
I do not guess it from the operating system or normalize names myself. I reject
unresolved identity, including dangling destination links, and stop if probe
creation, lookup or removal fails. A cleanup failure or killed compiler can
leave that empty probe directory at the output path. Existing destinations are
never replaced by a probe. This remains a stable-filesystem check, not atomic
publication or protection against concurrent path replacement.

My C-seed shadow JSON aggregates the selected graph. Completed runs report
`completed: true`, `test_count`, and failed tests with their owning
`source_file`. The recorded first line/column is not full cross-module
callee provenance.
Before starting the child I replace an old report with `success: false`,
`completed: false`, and a null count. A crash, early process exit or deadline cannot
leave a previous successful report masquerading as this run. I keep shadow
locals and runtime mutations out of the production compiler environment.
My filesystem and module-cache bootstrap shadows create private fixtures;
they do not depend on a particular working directory or fixed temporary names.

My C-seed shadow runner
records failed foreign dispatch even when the test ignores its result, and
includes the call location in shadow JSON. I distinguish this failure from a
successful void return. My interpreter uses libffi for fixed-arity scalar and
pointer signatures, including floating-point arguments and results. I reject
array arguments and non-opaque aggregate signatures. My native declarations
remain trusted: I cannot infer a C function's signature from its address.
Qualified calls, returned-function calls and function variables now share
checked foreign dispatch. Native builtin callbacks inherit their invoking
source call for shadow failure reporting; host calls without a source node do
not invent a location. This interpreter call context is sequential, not a
reentrant or concurrent execution guarantee. I test floating-point calls and
ignored missing-symbol failures through these routes, including `map`.
General interpreter error propagation remains separate work; a successful
shadow is not an ABI safety guarantee.
My `test_imported_shadow_selection` characterizes direct and transitive pure
imports: all three drivers reject a failing dependency shadow by default and
allow the explicit root-only opt-out. I test diamond imports, symlinks,
dependency-before-root execution and production separation. These tests do
not establish complete backend semantic equivalence.

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
