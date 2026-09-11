# Nano Logic

I compile a bounded Datalog subset: integer facts, Horn rules, ground
queries, and a deterministic least fixed-point. Unification of integers
is verified NanoISA (`lg_unify` / `I64_EQ`). Joins, projection, and
iteration stay in the host. I am not Prolog, not a Datalog product,
and not NSI policy compilation.

This is a C host (`src/logic/`), like `src/shell/`. There is no
`src_nano` twin. NanoLang stays my native language.

I do not use choice points or tabling. Naive iteration over a finite
EDB is enough for this subset. Recursion is allowed; derivation caps
fail closed.

## Restricted policy profile

`grant` and `allow` are ordinary predicates. `query allow 7` is a
ground policy query. I do not ingest NSI effect maps or deployment
documents. That is the documented restricted profile.

## What I compile

```
<program>  ::= <item>+
<item>     ::= fact <name> <int>+
             | rule <name> <var>+ :- <atom> (, <atom>)*
             | query <name> <int>+
<atom>     ::= <name> <term>+
<term>     ::= <int> | <var>
```

Arity is 1–3. Tuples are unique. Rules fire in source order. Tuples
are stored in lexicographic order, so the fixed-point is deterministic.

## Pinned tests

`make test-logic` is the gate. The cases in `tests/logic/test_logic.c`
are the pinned subset.

Set `NL_LG_TRACE=1` to print the emitted assembler.

## Intentional exclusions

- negation, aggregation, and function symbols
- choice points and tabling
- open (non-ground) queries
- NSI policy documents as source
