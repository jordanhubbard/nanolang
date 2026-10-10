# My generic result fix and global-storage audit

I retain the concrete record result name on a checked generic call. Previously,
`identity(identity(NSType {...}))` passed a record to an emitted
`identity_T(void*)` specialization. I copy the substituted record name into the
call's existing owned result metadata, so enclosing calls and field projections
use its actual type. Primitive generic results clear record metadata.

`native-generic-pass.log` passes literal, local, ordinary returned-record and
nested generic arguments, direct result-field access and two distinct record
types. The same fixture includes mandatory primitive and record shadows. The
existing `tests/unit/test_generics.nano` also compiles and executes; its only
output is the existing unused-parameter warning in `primitive-generic.log`.
`native-generic-refusal.log` retains the pre-fix C compilation failure.

## My remaining generic NanoISA producer gap

This native C fix does not establish generic NanoISA production. The retained
`generic-producer-repro.nano` fails in both public producers: NanoVirt lowers
one generic identity body with a STRUCT result and its integer shadow fails
before publication; the self-hosted compiler rejects primitive and record
arguments against literal `T`. I retain both diagnostics. I require consistent
checked specializations through checking, lowering, signatures and mandatory
shadows; I do not replace this with unchecked dynamic result admission.

## My current global-storage evidence

I inspected the actual implementation and found integrated functionality behind
historical scope rows describing an old native refusal. At base `370953ae6`,
all 28 imported-global methods pass through both installed compiler stages
(`installed-global-tests.log`), and all 28 pass separately through the C
frontend (`cseed-global-tests.log`). They execute generated modules in NanoVM
and standalone C AOT with address/undefined sanitizers, and preserve prior
outputs on negative cases. They cover module identity, aliases, mutation,
strings, arrays, maps, records, closures, callee evaluation order, dependency
initialization and resource/type refusals.

`record-retention-tests.log` additionally passes all four record-global tests,
including nested owned strings/arrays, 4,096 global overwrites with forced
collections, direct/global/tail-call roots and invalid projection/argument
refusals. These establish current local implementation, not exact-candidate
Linux/Darwin acceptance. The full release/platform gates remain open.
