# Nano Object

I compile a bounded Smalltalk-like subset to verified NanoISA methods
and dispatch them in one host process. I am not Smalltalk, Self, or an
image-based IDE. I am a 4.6 laboratory frontend: message send, object
identity, mutable slots, reflection, live method replacement, layout
growth, callable handles, and a host-side inline cache. I do not add
cache-specific opcodes.

This is a C host (`src/object/`), like `src/actor/`. There is no
`src_nano` twin. NanoLang stays my native language. Method bodies are
NanoISA. Identity, slots, caches, and the snapshot stay in C.

## What I compile

```
<program>     ::= <class>* <main>
<class>       ::= class <Name> { <id>* <method>* }
<method>      ::= method <id> <id>* { <assign>* <expr>? }
<assign>      ::= <id> = <expr>
<main>        ::= main { <stmt>* <expr>? }
<stmt>        ::= <id> = new <Name>
                | <id> = send <id> <id> <expr>*
                | send <id> <id> <expr>*
                | <id> = handle <Name> <id>
                | sendvia <id> <id>
                | replace <Name> <id> { <assign>* <expr>? }
                | extend <Name> <id>
<expr>        ::= <int> | <id> | ( <expr> ) | <expr> +|- <expr>
                | classof <id> | slots <id>
```

Every slot is `int`. An object id is an `int`. Methods always receive
eight slot locals so `extend` can grow a class without changing
portable arity. A method returns a 9-tuple `(result, slot0..slot7)`.
The host writes the slots back. That is mutation. It is not a new
opcode.

Each `send` site keeps a host inline cache: class id plus function
index. A hit uses the cached index. A miss looks up the selector.
Those counters are not NanoISA.

`replace` compiles a second function and switches that selector on
the class. `extend` adds a zero slot. `handle` / `sendvia` call a
function index without naming the selector again.

`nl_object_last_image` is a text snapshot of live objects after eval.
That is the image for this subset, not a Smalltalk image file.

## Pinned tests

`make test-object` is the gate. The cases in
`tests/object/test_object.c` are the pinned subset.

Set `NL_OBJ_TRACE=1` to print the emitted assembler.

## Intentional exclusions

- cache-specific portable opcodes
- a Smalltalk image format or browser
- blocks, doesNotUnderstand, and become:
- more than eight slots
