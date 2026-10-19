# My checked call-borrow component qualification

I implement the reviewed invocation-scoped ledger contract for #998. These are
local component results, not a fresh raw Stage 1/Stage 2 or platform receipt.
My final helper-shadow builds and focused provider/source qualification pass.
Fresh aggregate raw generations and Linux/Darwin qualification remain open.

I retain actual checker events, never runtime authority. C owns copied snapshots
through Environment teardown. Nano carries invocation-local context from its
successful final checker pass to a private codegen bridge. The public borrowed
lowerer returns only untrusted site descriptors; exact owner/epoch/site checks
precede publication of independent `nano.borrow-place.v1` advisory records.

## Completed local controls

- My ledger provider passes lifecycle, stale identity, capture budget and all
  three observed allocation-denial positions with independent recovery.
- My actual Environment destructor passes active-scope cleanup, all three
  allocation-denial positions and recovery. The harness wraps only ledger
  allocations; I do not claim a whole-program leak census. Linking the real
  common providers initially required the same test-owned `g_argc/g_argv`
  globals as the existing environment harness; no production stub is used.
- My expanded advisory codec passes 1250 checks, including copied transport,
  canonical reconstruction, malformed/duplicate refusal and executable
  invariance under stripped/forged metadata.
- My C-hosted compiler builds successfully after two private-module visibility
  failures and a generated-C missing direct import. I retain those failures.
- Two paired source methods pass in 1.338s: exact root/nested events, canonical
  roundtrip, actual VM and sanitized native execution, advisory tampering
  invariance and prior-output preservation on checking/runtime-shadow failure.
  An initial refusal fixture used reserved `use`; a diagnostic assertion caught
  the parser refusal before I corrected that test name.
- My actual changed-Nano checker-context helper passes in 71.371s. It checks raw
  absence, real event lookup, program/shadow descriptor lifetime, checked
  publication, stale old-context rejection after rechecking and later raw
  absence. Its first test source had `List<T>=` spacing errors, retained here.
- My four original scalar borrow fixtures pass in 3.975s across C, raw Nano
  emission and C-hosted canonical Nano: 12 actual VM/sanitized native artifacts.
  The comparator removes only validated, exclusively owned trailing advisory
  strings. Three adversarial comparator controls preserve CODE/authority and
  reject shared references, duplicate sites and unknown reference schemas.

## Existing import boundary

My initial two-module positive probe exceeds the existing source borrow profile
of main plus one borrowed helper; both producers refuse it. I preserve this
refusal rather than widening executable admission for a metadata task.

For the narrower imported Counter/update fixture, C accepts and records the root
main call's actual source coordinates. Changed Nano and the aggregate's baseline
Stage 1 both refuse the exact imported helper profile. My bounded tests preserve
that known difference and prior output. I do not claim paired imported-call fact
coverage from this checkpoint. The captured C ledger's owning-file field and
private identity checks remain distinct from unsupported source profiles.

## Final frozen component receipt

My final changed C provider, C-hosted canonical compiler and raw Nano emitter
all build successfully, including the added helper shadows. The combined source
and comparator corpus passes **9 methods in 77.384s**. It covers the actual
checker-context helper, paired root/nested facts, copied and canonical transport,
VM and ASan/UBSan native behavior, advisory tampering invariance, all four original
scalar-borrow fixtures across three emission routes, retained import boundaries,
and prior-output preservation on later checking/runtime-shadow failure.

My final actual Environment harness passes active-scope destruction, all three
observed ledger allocation failures and independent recovery. Actual environment
index, generated-list metadata and standalone resource-flow allocation neighbors
also compile, link and pass against the final providers.

`final-provider-sha256.json` pins those exact binaries.
`nanolang-borrow-final-inputs.json` pins all 22 frozen source/test/seed inputs;
I checked every hash after the source suite. The final strict GCC 16 pass then
found one misleading-indentation warning in `same_place`. I retained its failure
and split the existing three statements onto separate lines without changing
expressions or control flow. All **14 GCC/Clang strict provider syntax checks**
and a rebuilt direct ledger lifecycle/limits/allocation harness then passed.
`final-reviewed-source-sha256.json` pins that final formatting correction; the
source/native receipt remains explicitly pinned to the pre-format compiler.

The initial logs and pins remain historical evidence, including corrected
visibility, parser-spacing and fixture-profile failures. I do not substitute
these C-hosted checks for the required fresh raw Stage 1/Stage 2 build: the new
checker/result record shapes must still pass aggregate translation, fixed point,
and exact Linux/Darwin gates. I leave #998 open.

## My link-consumer audit

I add the ledger to `COMMON_SOURCES`. The generated-list metadata test rebuilds
poisoned `env.c` but keeps the rest of `COMMON_OBJECTS`; the environment symbol
index test includes `env.c` and filters out only `env.o`. Both retain the ledger
provider. The service-input sanitizer similarly replaces selected providers
without removing the ledger. The FileServiceParser harness rebuilds its listed
lexer/UTF8/environment/nominal/interpreter/transpiler providers, while its Make
caller supplies the remaining common/runtime objects including the ledger.
The standalone resource-flow allocation harness includes `resource_flow.c`
directly; I explicitly add the ledger source to that recipe. My own Environment
backstop harness replaces only the ordinary ledger object with its allocation-
observed provider. This audit does not replace the final aggregate parser and
platform checks.
