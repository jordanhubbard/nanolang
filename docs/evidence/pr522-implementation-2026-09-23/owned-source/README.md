# My canonical owned-union source checkpoint

I retain concrete resource-record and nested-union payload identities in my
self-hosted source producer. Children precede their parents in ownership format 4;
resource flags follow stored payloads. My constructor temporaries preserve source
evaluation order and transfer fields in declaration order. Aliases, arguments and
results move whole owned unions. Complete selected patterns emit
`OWN_UNPACK_VARIANT`; empty and ordinary arms discharge the outer obligation.
I retain ownership checks across branch joins and block-valued match results.

My affine verifier now permits a selected scalar/STRING observation of a rooted
union. Unknown variants and resource-field projection remain refused. This permits
ordinary sibling arms to inspect a scalar without copying a resource payload.

## My passing checks

- Fresh `make bootstrap`: both compiler stages, hello executions, installed
  compiler publication and C-seed independence pass. This is not a final-source
  fixed-point comparison.
- The unchanged generic selected ownership, selected patterns and generic identity
  suites pass all 44 methods across `nanoc_c`, `nanoc_stage1` and `nanoc_stage2`.
  The native compilers run mandatory shadows before publishing each accepted case;
  rejected cases retain their original prior-output checks.
- `make test-owned-union-source`: ten accepted original sources execute through raw
  bytecode and native C with ASan/UBSan/leak checks. Nine original refused sources
  preserve previous bytecode output. The guarded-match diagnostic remains tested
  in the original three-compiler suite rather than duplicated in this raw driver.
- Two imported scalar-union source regression methods pass, including identity and
  shadow refusals. The scalar-union raw runtime passes.
- Complete native regression: 1,500 shape assertions and 2,428 translator assertions.
- Selected ownership analysis: 837 normal and 1,199 allocation-fault bytecode checks;
  affine state remains 425/457. These pass scoped ASan/UBSan with leak detection,
  instrumenting the state, bytecode, ownership and verifier units plus fixtures.
  Other linked dependencies remain ordinary objects.
- The owned-union runtime passes 1,443 fixture checks and 81 VM allocation checks,
  now also observing selected scalar/STRING payloads. Schema checks pass all 33 tests.
- The repository shadow-presence check and `git diff --check` pass.

## My remaining boundaries

I retain the broader selected-variant suite: 15 of 16 methods pass across the three
compilers. `test_scrutinee_call_evaluates_once` passes the C seed but fails both
native stages because my owned source profile refuses its mutable scalar global
counter. I preserve its exactly-once assertions in task
`task_6fa28259d6c676a11406fd124a7d28f9`; integration waits on that task as well as
remaining generic frontend parity.

A freshly linked C-based `nano_virt` refuses the original resource/ordinary-arm
case before shadow publication: its separate `src/nanovirt/borrow_codegen.inc`
still requires scalar concrete union arguments. The successful self-hosted source
producer does not qualify that C frontend. The recorded command has exit code 1.

Hosted run `35908912592` belongs to parent head `74b59b07d`; its checkout merge
commit is recorded in the worker result. Units-00 passes its initial bootstrap,
then its target suite rebuilds Stage1 and stops compiler shadows at 60 seconds.
I retain the failed terminal and worker result. The complete hosted run, other
known compatibility failures and release gates remain unqualified. I do not label
the timeout infrastructure or remove the failed worker.

`provenance.json` records source and executable hashes. `logs.json` records the
uncompressed hashes of retained logs. PR #522 stays draft; these local results do
not establish release readiness.
