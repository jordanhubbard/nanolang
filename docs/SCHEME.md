# Nano Scheme

I compile a bounded Scheme subset to verified NanoISA. I am not a
Scheme report implementation. I am a 4.6 laboratory frontend: lexical
scope, closures, first-class procedures, pairs, interactive evaluation,
and proper tail calls for named recursion. Continuations, macros,
assignment, and ports stay out of scope.

This is a C host compiler (`src/scheme/`), like `src/forth/`. There is
no `src_nano` twin. NanoLang stays my native language.

## What I compile

```
<program>     ::= <form>*
<form>        ::= <define> | <expr>
<define>      ::= (define (<name> <arg>*) <expr>+)
                | (define <name> (lambda (<arg>*) <expr>+))
<expr>        ::= <integer> | #t | #f | () | <name>
                | (quote <datum>) | '<datum>
                | (if <expr> <expr> <expr>)
                | (begin <expr>+)
                | (lambda (<arg>*) <expr>+)
                | (let ((<name> <expr>)*) <expr>+)
                | (cons <expr> <expr>) | (car <expr>) | (cdr <expr>)
                | (null? <expr>) | (pair? <expr>)
                | (+ <expr>*) | (- <expr>+) | (* <expr>*) | (/ <expr> <expr>+)
                | (= <expr> <expr>) | (< <expr> <expr>) | (> <expr> <expr>)
                | (eq? <expr> <expr>) | (not <expr>)
                | (<expr> <expr>*)
```

`let` desugars to an immediately applied lambda. `if` without `else`
becomes `(if test then #f)`. Only `#f` is false; `0` is true.

`'()` is a 2-tuple of voids so a list-returning procedure always
returns `tuple`. A pair is a 2-tuple. Closures return as `function`
(the VM accepts `TAG_CLOSURE` there). Arithmetic returns `int`.

Named self-recursion in tail position emits `TAIL_CALL`. I checked
`(sum 10000 0)` at constant frame depth. First-class `CALL_INDIRECT`
in tail position still returns; that is not a proper tail call.

## Interactive evaluation

`nl_scheme_open` keeps a session of `define` forms. A later `define`
of the same name replaces the previous body. That is live code
publication for this subset, not an image or `eval` of data as code.

## Pinned tests

`make test-scheme` is the gate. The cases in
`tests/scheme/test_scheme.c` and `tests/scheme/r5rs_pin.scm` are the
pinned subset. They follow examples from R5RS (integers, `if`,
`lambda`, `define`, `cons`). I do not run a third-party Scheme suite.

## Intentional exclusions

- `call/cc` and dynamic-wind
- `set!`, boxes, and other assignment
- macros, `syntax-rules`, quasiquote
- strings, vectors, characters, ports
- rest arguments, multiple values
- `eval` of lists as code
- the full numeric tower
- a Scheme report banner

I will look at continuations only after this closure subset and
exception semantics are stable. They are not stable yet, so I refuse
`call/cc`.

## Contract

I emit `.nvm` v2, attach DEBUG entries, and pass `nl_frontend_accept`
with `NL_FE_SCHEME`. Opcodes are the shared ISA. See
`docs/NANOISA_FRONTEND.md`.
