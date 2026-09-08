# Frontend matrix

I keep NanoLang as my native language. Forth, Scheme, ML, Actor,
Dataflow, Object, Shell, and Logic are bounded architecture probes.
They emit the same verified `.nvm` v2 and they pass the same verifier.
They do not become product compilers because this table exists.

`make test-frontend-matrix` is the gate. It runs equivalent integer
fixtures, accepts one shared library from NanoLang, Forth, Scheme, and
ML, starts a supervised Actor from Shell only after a Logic policy
query allows it, and prints compile time, module size, instruction
mix, interned-string count, call count, and eval time.

## How each frontend exercises NanoISA

| Pressure | NanoLang | Forth | Scheme | ML | Actor | Dataflow | Object | Shell | Logic |
|---|---|---|---|---|---|---|---|---|---|
| Typing | explicit | typed words | dynamic | Hindley–Milner | int messages | int streams | slots | structured i64 | ground ints |
| Calls | prefix | colon / CALL | CALL_INDIRECT | app / CALL | send | node fire | send | pipe | query |
| Closures | native | — | first-class | fun/fn | — | — | handle | — | — |
| Stacks | VM | data/return | VM | VM | per-actor VM | VM | VM | VM | unify |
| Matching | match | — | cond | exhaustive case | receive | — | — | — | unify |
| Concurrency | — | — | — | — | mailboxes | DAG | — | cancel | — |
| Services | NSI + CALL_MODULE | NSI + CALL_MODULE | accept shared lib | accept shared lib | supervised child | — | — | service hook | policy query |
| Replacement | — | — | session define | — | replace | — | replace | — | — |
| Replay | — | — | — | — | restart | interned feeds | image | — | deterministic FP |

A dash means that frontend does not exercise the pressure in the
pinned subset. It is not a defect in the ISA.

## Shared library

The `add` library in `tests/frontend/test_frontend_matrix.c` is the
same assembler Forth and NanoLang already share. Scheme and ML accept
that verified module too. They do not grow a second linking story.

## Supervised service

Echo is a supervised Actor. Logic `query allow 1` is evaluated first.
Shell `need service` / `service 1` starts Echo only when a test binds
`nl_shell_set_service` to `nl_actor_eval_i64` of that program. Policy
is not in the Actor source. I still do not administer Phase 18
service graphs.

## Measurements

The test prints one row per frontend. Times may be zero on a fast
host; module `code_size` and instruction count must be nonzero. I do
not claim a portable allocation profiler. Interned string count is the
published allocation proxy.
