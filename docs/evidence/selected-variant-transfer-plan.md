# Selected variant ownership: implementation order

I track this work as `task_c17b55115379414980609a5d867ccad1`. My fixed union
collection rejection is already integrated. It does not transfer a selected
payload.

My initial ordinary language example uses `let Choice.Some { owner, label } =
payload` inside an exhaustive match. Both parsers rejected the qualified
pattern before ownership checking. I preserve that initial program and the
three compiler logs in `/tmp/nanolang-selected-variant-baseline/`.

I implement these dependencies in order:

1. Preserve qualified pattern spelling in both parsers. Validate the complete,
   unique field set against the actual selected nongeneric variant. Ordinary
   payloads establish parsing, checking, interpreter shadows and native storage.
2. Retain a selected variant identity in each ownership flow. Resolve field
   identities through that variant, including nested resource records. An
   empty or ordinary arm does not acquire another arm's resource obligation.
3. Move an exhaustive, unguarded nongeneric scrutinee exactly once before arm
   flows split. Transfer each selected payload through the complete pattern,
   resolve all resource fields on every exit, then join outer ownership states.
4. Check positive execution and rejection parity on the C seed and both
   self-hosted stages. Include unresolved payloads, incomplete fields, repeated
   uses, original scrutinee reuse, terminating exits and incompatible joins.

Until steps 2–4 pass, my existing resource-match diagnostic remains active.
I retain conservative collection, generic, tuple and borrow boundaries.
Guarded matches and wildcard ownership shortcuts need separate acceptance;
ordinary native execution does not establish NanoISA ownership metadata.

## Qualified pattern checkpoint — September 17, 2026

My fresh three-stage bootstrap passes. Five methods in
`tests/test_selected_variant_patterns.py` pass across the C seed and both
self-hosted stages (15 compiler decisions, 8.158 seconds). Complete ordinary
selected payloads execute; missing, duplicate and wrong-variant patterns retain
an existing output artifact. The fifth method confirms that my resource-match
guard still rejects owned payload transfer. My 44 adjacent affine, generic and
fixed-collection methods pass in 75.774 seconds.

I lower a checked complete pattern to one inferred native temporary. A checked
empty pattern over an identifier has no fields and emits no runtime storage;
non-identifier initializers retain their evaluation. This is ordinary native
pattern acceptance, not completed ownership transfer or module-owned union
identity.

Local logs: `/tmp/nanolang-selected-pattern-bootstrap.log`,
`/tmp/nanolang-selected-pattern-all.log`, and
`/tmp/nanolang-selected-pattern-adjacent.log`.
