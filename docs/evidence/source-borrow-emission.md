# My first source-produced borrowed calls

I lower a closed resource-borrow profile from ordinary source through both
NanoVirt and my self-hosted NanoISA emitter. This is task
`task_5057848888b246f686fd2b8e48d2c19a`, under my open affine and call-scoped
borrow parents. My [source contract](../NANOISA_SOURCE_BORROWS.md) is narrower
than my native source language.

I retain every declared resource's exact nominal layout and int/bool field
tags. Entry is function 0; one borrowed helper is function 1. Constructor
fields evaluate once in source order, then enter `OWN_PACK` in declaration
order. A checked root borrow surrounds `CALL_REF` with an explicit region.
Destructuring moves the original owner; scalar-only shell disposal uses
`OWN_UNPACK_LOCAL` and scalar pops. The shared verifier checks the completed
module before bytecode publication.

I compile the complete selected shadow suffix into a synthetic entry and the
same helper. Assertions remain assertions. False assertions inside the helper
propagate failure and clean actual caller ownership. My synthetic name avoids
a user helper with the same spelling. I refuse a selected graph that exceeds
my profile, including an empty/scalar-only entry without an actual ownership
transfer: my existing owned verifier requires that transfer. I do not add a
dummy owner or relax its rule.

## My measured checks

At source checkpoint `4eb77ac6`:

- A fresh default three-stage native bootstrap passed.
- `tests.test_source_borrow_emission`: five methods passed in 134.578 seconds.
  Four fixtures cover repeated shared reads, caller-visible exclusive updates,
  Boolean helper results, and constructor field evaluation order with a
  mutating borrowed call. Twelve complete disassembly comparisons match the
  C-seed artifact against emitters built by C seed, Stage 1 and Stage 2.
- Those fixtures execute as 24 ordinary VM/native pairs, including canonical
  Stage 1/Stage 2 `--emit-nvm` publication. Fifteen additional native executions
  cover selected shadows, false entry/helper assertions, suffix selection and
  synthetic-name collision. All 39 generated native executions use strict C
  warnings plus ASan/UBSan and unsuppressed leak detection.
- Eight refusal/failure cases across NanoVirt and both canonical stages retain
  the previous output: false shadow/helper assertions, scalar-only selected
  entry, a shadow calling main, an extra function, a loop, a heap field and a
  nominal mismatch. The explicit empty suffix also refuses in all three
  compiler-built shadow emitters.
- The unchanged shared/exclusive/nested source suites passed all 25 methods
  in 91.527 seconds.

My logs are `/tmp/nanolang-source-borrow-bootstrap-r2.log`,
`/tmp/nanolang-source-borrow-paired-r2.log` and
`/tmp/nanolang-source-borrow-adjacent-source.log`. The earlier paired log
retains my test's incorrect expectation that an empty owned entry would be
admitted. The corrected contract keeps the verifier's refusal and adds a
positive suffix with actual owner transfer.

## My integrated checks

I restacked through main `1686172a` (PRs #593, #595, #597, #598, #600, #596 and
#602), preserving their optional metadata, native and compiler changes.
My specialized producer sources are unchanged across the final restack.
The integrated three-stage bootstrap passed before the final C-only name
allocation checks; the later main changes did not change my self-hosted
compiler source. At final source `251abb08`:

- The existing NanoISA gate passed 86 exact comparison checks and 88 methods
  (116.235 seconds). My integrated source suite passed all five methods
  (135.897 seconds), and the unchanged borrow suites passed 25 methods
  (92.834 seconds).
- I then checked C name-interning failures explicitly and strengthened false
  assertion tests to require the exact process status. The retained first
  attempt expected internal status 2; my existing standalone wrapper maps
  that status to process exit 1. Both corrected assertion methods passed
  in 132.952 seconds. I changed no runtime behavior and accept no signal
  termination as an expected assertion failure.
- After the final restack I rebuilt the current tools. All four fixtures
  produce equal complete dumps through NanoVirt, the raw self-hosted emitter
  and both canonical stages: 16 modules verify and run in VM. Their 32
  standalone native runs pass strict GCC/Clang compilation with
  ASan/UBSan/unsuppressed LSan. My Clang is Ubuntu Clang 18.1.3.
- Current `test-nanovirt` passed 89 checks; `test-owned-assertions` passed
  its five artifacts and 959 lifecycle checks.

My integrated logs are `/tmp/nanolang-source-borrow-integrated-bootstrap.log`,
`/tmp/nanolang-source-borrow-emitter-gate.log`,
`/tmp/nanolang-source-borrow-integrated-paired.log`,
`/tmp/nanolang-source-borrow-integrated-adjacent.log`,
`/tmp/nanolang-source-borrow-exact-status.log`,
`/tmp/nanolang-source-borrow-tip-positive.log` and
`/tmp/nanolang-source-borrow-tip-core.log`. The standalone-status expectation
failure remains in `/tmp/nanolang-source-borrow-final-paired.log`.

## My remaining boundaries

I do not admit imports, globals/initializers, branches/loops, deeper or
recursive calls, multiple borrowed parameters, reference escapes, nested
resource paths, heap fields, U8 root projection, or aggregate results in this
source profile. Unsupported selected shadows refuse the entire canonical
publication. My raw assembly API is a lowering API; mandatory supervision is
a separate canonical driver responsibility, exercised above. Ordinary native
borrow language remains broader and its existing tests are retained.

My specialized ownership descriptors identify physical local slots exactly.
The optional ordinary-local name annotations from PR #593 do not yet annotate
this new specialized path. That continuation is
`task_0178be8daf554a69969144b6657e7756`, under reconstruction parent
`task_4bd034f6029b7458201db74e2c3aeb32`. Record, field and function names remain retained.
I do not infer complete affine/source admission, scheduler behavior, compiler
correctness or release readiness from these tests.
