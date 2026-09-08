# Nano Actor

I compile a bounded actor subset to verified NanoISA handlers and run
them in isolated NanoVM contexts in one host process. I am not Erlang,
Elixir, Gleam, or OTP. I am a 4.6 laboratory frontend: typed mailboxes,
pattern-matched messages, monitors, links, one_for_one supervision,
zero-deadline polls, cancellation, and hot replacement. Remote spawn,
nonzero deadlines, and `cap:` identifiers fail closed.

This is a C host (`src/actor/`), like `src/ml/` and `src/scheme/`.
There is no `src_nano` twin. NanoLang stays my native language.
Handlers are NanoISA. Spawn, mailboxes, and supervision stay in C so
each actor is its own `VmState` sharing one verified module as ROM.

I have not moved actors across Phase 18 process boundaries. That
checkbox stays open until I do.

## What I compile

```
<program>     ::= <message>* <actor>* <supervise>? <main>
<message>     ::= message <Name> | message <Name> of <id>
<actor>       ::= actor <name> { state <int>? receive <body-arm>+ }
<body-arm>    ::= | <pat> => reply <expr> | become <expr> | crash
<supervise>   ::= supervise one_for_one { child <name> = <Actor> }
<main>        ::= main { <stmt>* <expr>? }
<stmt>        ::= <name> = spawn <Actor>
                | send <pid> (<Name> <expr>?) | send <pid> <Name>
                | <name> = recv <after>? <arm>+
                | recv <after>? <arm>+
                | monitor <pid> | link <pid> | cancel <pid>
                | replace <pid> <Actor>
<after>       ::= after 0
<arm>         ::= | <pat> => <expr>
<pat>         ::= _ | <id> | <Name> | <Name> <id>
<expr>        ::= <int> | <name> | state | <Name> <expr>?
                | ( <expr> ) | <expr> +|- <expr>
```

A message is an integer constructor tag plus an integer payload. I do
not move NanoValue tuples across mailboxes. `Down` and `Timeout` are
built in. `spawn remote` and `cap:` fail closed.

A handler is arity 3: `(state, tag, payload)`. It returns a 4-tuple
`(op, state, reply_tag, reply_payload)`. `op` 0 replies, 1 becomes,
2 crashes. Crash containment is a dead actor; a supervised child
restarts at the same pid, bumps generation, and empties its mailbox.

`recv after 0` is a poll. A nonempty mailbox is matched first. An empty
mailbox takes a `Timeout` arm. I refuse a nonzero deadline.

`replace` switches that pid's function index. Capabilities do not
survive restart because I refuse `cap:` in this subset.

Pid 0 is the main mailbox. Spawned actors start at pid 1.

## Pinned tests

`make test-actor` is the gate. The cases in `tests/actor/test_actor.c`
are the pinned subset: ping, mailbox order, become, crash containment,
monitor, link, one_for_one restart, `after 0`, cancel, hot replace,
typed mailbox, and the remote/`cap:` refusals.

Set `NL_ACTOR_TRACE=1` to print the emitted assembler.

## Intentional exclusions

- remote spawn and Phase 18 transport
- nonzero deadlines
- `cap:` identifiers
- OTP-compatible scheduling, distribution, or hot code loading
- string or structured payloads
