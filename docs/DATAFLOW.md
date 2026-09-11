# Nano Dataflow

I compile a bounded dataflow subset to verified NanoISA node bodies and
run the graph in one host process. I am not a workflow engine, Spark, or
a streaming product. I am a 4.6 laboratory frontend: typed nodes,
bounded streams, backpressure, explicit effects, provenance, replay,
cancellation, retries, and parallel determinism. `place remote` fails
closed. I have not mapped graphs onto Phase 18 service-process
boundaries.

This is a C host (`src/dataflow/`), like `src/actor/`. There is no
`src_nano` twin. NanoLang stays my native language. Node bodies are
NanoISA. Scheduling, buffers, journals, and retries stay in C.

## What I compile

```
<program>     ::= <node>* <buffer>? <graph> <main>
<node>        ::= node <name> <id>* <retry>? <effect>? = <expr>
<retry>       ::= retry <int>
<effect>      ::= effect IO | effect Err | effect State
<buffer>      ::= buffer <int>
<graph>       ::= graph { <wire>* }
<wire>        ::= <name> = in
                | <name> = <node> <name>*
                | out <name>
                | place <name> local
                | place <name> remote
<main>        ::= main { <stmt>* }
<stmt>        ::= feed <name> <int> | drain | cancel <name>
<expr>        ::= fail
                | if <expr> then <expr> else <expr>
                | <expr> +|-|*|/|== <expr>
                | <int> | <id> | attempt | ( <expr> )
```

Every stream is `int`. A node is a function of its named inputs plus
`attempt` (0-based retry count). It returns a 2-tuple `(ok, value)`.
`fail` is `ok = 0`. A node with `retry N` re-invokes the same inputs
while `ok = 0` and `attempt < N`.

`buffer N` is the capacity of every edge. A `feed` into a full buffer
fails closed. That is backpressure. I do not fire nodes until `drain`.

`drain` fires ready nodes until the sink has a value. Ready-set order
is lowest instance id, or highest when I schedule in reverse. Those
two orders must agree.

External inputs are the `feed` list. I intern each as
`.string "feed <port> <value>"` so a completed run names every input
needed to reproduce it. Evaluating the same source twice is replay.

`place remote` fails closed. `place local` is a no-op. Cancellation
marks an instance; firing it fails closed.

A bounded bulk payload in this subset is several integer feeds into
one node, copied along edges. I do not map NSI shared memory here.

## Pinned tests

`make test-dataflow` is the gate. The cases in
`tests/dataflow/test_dataflow.c` are the pinned subset.

Set `NL_DF_TRACE=1` to print the emitted assembler.

## Intentional exclusions

- remote / service-process placement (Phase 18)
- NSI shared-memory maps
- cycles, windows, and event-time
- non-integer stream types
- changing results when the scheduler order changes
