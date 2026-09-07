# Nano ML

I compile a bounded ML-family subset to verified NanoISA. I am not
Standard ML or OCaml. I am a 4.6 laboratory frontend: static inference,
algebraic data types, exhaustive pattern matching, immutable values,
higher-order functions, and a `signature` check. Refs, exceptions,
assignment, functors, and mutual `and` stay out of scope.

This is a C host compiler (`src/ml/`), like `src/scheme/` and
`src/forth/`. There is no `src_nano` twin. NanoLang stays my native
language.

## What I compile

```
<program>     ::= <decl>* <expr>?
<decl>        ::= <datatype> | <signature> | <fun>
<datatype>    ::= datatype <name> = <ctor> (| <ctor>)*
<ctor>        ::= <Name> | <Name> of <type>
<signature>   ::= signature <Name> = sig <spec>* end
<spec>        ::= val <name> : <type>
<fun>         ::= fun <name> <id>+ = <expr>
<expr>        ::= <int> | true | false | <name> | <Name>
                | ( <expr> , <expr> ) | ( <expr> )
                | <expr> <expr>          (* same line *)
                | <expr> +|-|*|/|=|<>|<|> <expr>
                | if <expr> then <expr> else <expr>
                | fn <id> => <expr>
                | let val <id> = <expr> in <expr> end
                | case <expr> of <arm> (| <arm>)*
<arm>         ::= <pat> => <expr>
<pat>         ::= _ | <id> | ( <pat> , <pat> ) | <Name> <pat>?
<type>        ::= int | bool | <name> | <type> * <type> | <type> -> <type>
```

`fun add a b = a + b` is curried: it is `fn a => fn b => a + b`.
Application is juxtaposition on one line. A newline ends an application
so a following `fun` or trailing expression is a new phrase, not an
argument.

`if` takes a `bool`. Constructors are capitalized. An algebraic value
is a 2-tuple `(tag, payload)`. A nullary constructor uses a void
payload so the result is still `tuple`. Closures return as `function`.
Arithmetic returns `int`.

A `case` must be exhaustive: every constructor, or a variable / `_`.
I refuse an incomplete `case`. Pair patterns are exhaustive when one
arm is a pair, a variable, or `_`.

A `signature` names values and types. After I infer a matching `fun`,
I instantiate its scheme and unify it with the spec. A mismatch fails
closed.

`let val` is non-recursive. I generalize at that binding and at each
top-level `fun`. Recursion through the function's own name is
monomorphic while I infer the body.

## Inferred types and exhaustiveness

`nl_ml_type_of` prints schemes without a leading `'`: `id` is `a -> a`.
Those strings are interned in the compiled module's string pool. I pass
`exhaustiveness = 1` on `NlFrontendFacts` because I do not emit a
module that still has an inexhaustive `case`.

## Shared aggregates

`fun fst p = case p of (a, b) => a` is function index 0 (`ml_fst`). A
NanoLang-labeled assembler module may `CALL_MODULE 0 0 1 1` after
building a 2-tuple. `make test-ml` checks that link with
`nl_frontend_accept_linked`.

## Pinned tests

`make test-ml` is the gate. The cases in `tests/ml/test_ml.c` are the
pinned subset. I do not run a third-party SML or OCaml suite.

Set `NL_ML_TRACE=1` to print the emitted assembler.

## Intentional exclusions

- `ref`, `exception`, and `:=`
- mutually recursive `and`
- functors, `struct`/`end` implementations, opaque types
- parametric datatypes (`'a option`); `option` here is monomorphic
- `call/cc`, mutable fields, and objects
- strings, reals, the full numeric tower
- a Standard ML or OCaml banner

I will look at functors and parametric datatypes only after this
inference and matching subset is stable. It is stable enough to emit
verified NanoISA. It is not a language product.

## Contract

I emit `.nvm` v2, attach DEBUG entries, intern inferred schemes, and
pass `nl_frontend_accept` with `NL_FE_ML`. Opcodes are the shared ISA.
See `docs/NANOISA_FRONTEND.md`.
