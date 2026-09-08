# Nano Shell

I compile a bounded orchestration subset to verified NanoISA functions
and run typed integer pipelines in one host process. I am not bash, not
a POSIX shell, and not the administrative language for service graphs.
I am a 4.6 laboratory frontend: structured `int` values in pipes, an
explicit `parse` adapter from text, and host effects only through
`need` capabilities. Missing a capability fails closed. A granted
capability still refuses real files, processes, networks, services,
streams, and remote execution in this subset.

This is a C host (`src/shell/`), like `src/object/`. There is no
`src_nano` twin. NanoLang stays my native language.

I do not administer Phase 18 service graphs. That checkbox stays open
until capability and policy enforcement are the ones the fabric
already tests, and this shell actually drives them.

## What I compile

```
<program>     ::= <need>* <fn>* <main>
<need>        ::= need files | need proc | need net | need svc
                | need stream | need remote
<fn>          ::= fn <name> <id>* = <expr>
<main>        ::= main { <stmt>* }
<stmt>        ::= <id> = <pipe> | cancel | <pipe>
<pipe>        ::= <expr> ( | <name> <expr>* )*
<expr>        ::= <int> | "<text>" | parse <expr> | <id>
                | read <expr> | run <expr> | connect <expr>
                | service <expr> | stream <expr> | remote <expr>
                | <expr> +|- <expr> | ( <expr> )
```

A pipe carries `int`. `"3" | add 1` fails closed. `parse "3" | add 1`
is 4. `2 | add 3` is `add(2, 3)`.

## Pinned tests

`make test-shell` is the gate. The cases in `tests/shell/test_shell.c`
are the pinned subset.

Set `NL_SH_TRACE=1` to print the emitted assembler.

## Intentional exclusions

- text as the pipeline data model
- real host files, processes, sockets, or Phase 18 services
- using me to administer service graphs
