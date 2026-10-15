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

This native C fix does not establish generic NanoISA production. At my recorded baseline, the retained
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

## My C-producer specialization checkpoint

I now retain generic declarations as templates and create concrete NanoISA
signatures and bodies keyed by source-body identity and parameter types.
I compile discovered instances through a worklist, preserving imported owner
context, concrete local declarations and direct generic tail calls. I do not
mutate shared template declarations when substituting local types.

My focused suite is `tests/test_cseed_generic_functions.py` (also available as
`make test-cseed-generic-functions`). I execute primitive and distinct record
specializations, nested/transitive calls, typed locals, 5,000 recursive tail
calls and qualified/selective imports from distinct owners. I retain mandatory
source shadows, verify emitted modules and run NanoVM plus C AOT with address
and undefined-behavior sanitizers. My repeated-variable negative case rejects
different nominal records and preserves the previous output.

I keep the full producer requirement open. The self-hosted checker/emitter is
not fixed by this C-producer change. Aggregate generic substitutions, contextual
function-value specialization and broader generic metadata still require work;
this checkpoint admits direct scalar and declared-record arguments only.
I also still require an exact-candidate bootstrap and platform qualification.

My [combined current C-producer run](cseed-specialization-tests.log) passes all
43 methods: six generic methods and 37 adjacent imported-global, record-array
and canonical shadow methods. I used LLVM clang for sanitized native products.
