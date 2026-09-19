# My public C integer conversion evidence

I qualify task_3a0411f257e4437ca446d2bbec0e5ed6 and exact length prerequisite task_66f75e0c3cd74d2989d158237c7be305 at source/harness `1ceed53a1fe107a44bc1d86f8fe30c0ddc5a66f8`. My production checkpoints are `4269fa6d` (integer snapshots) and `c04ab8ea` (exact length typing). My [contract](../PUBLIC_C_INT_CONVERSION_CONTRACT.md) precedes production and records both independent reviews. My [machine evidence](public-c-int-conversion.json) retains 19 source/provider/harness/tool identities, two compiler identities and 12 log hashes. I verified every frozen identity after all gates terminated.

I replace the exact INT conversion GNU/static buffer with portable signed decimal formatting and the existing stable snapshot pool. Float and integer conversion share checked allocation/registration and exit cleanup; formatting remains operation-specific. Allocation and cleanup diagnostics now say scalar. I retain process-lifetime results and free them at exit; I do not claim early reclamation or a memory bound. Binding, declaration, checked signature and expression callee resolution precede builtin lowering.

My new six methods cover C99/C11 at O0/O2, signed endpoints, once evaluation, helper-name collisions, mixed integer/float aliases through globals/returns/300 loop iterations, generated allocation/registration failure and cleanup, exact type/arity refusal, first diagnostic retention, path/stream output preservation and later valid emission. I also check ordinary nested integer conversion of string length, empty strings and the existing first-NUL boundary. Generated programs and isolated API/helper controls use ASan/UBSan with Linux leak checking enabled by default; I do not disable it. Existing ordinary backend programs run under their unchanged harness and are not an additional sanitizer claim.

| Final gate | Result |
|---|---|
| GCC integer/length6 plus adjacent public C18 | 24 pass, 10.490s |
| Clang integer/length6 plus adjacent public C18 | 24 pass, 13.444s |
| Unchanged public C backend suite | 7/7 pass |

I preserve the first GCC/Clang harness failures: a namespace assertion matched the intentionally declared user function rather than only the private helper. Those runs each passed three methods and stopped the endpoint method before generated C compilation. I corrected the assertion and froze again. Those corrected integer4+4 and adjacent18+18 gates passed, while the unchanged backend suite passed6/7 and checked-refused `int_to_string(str_length msg)`: existing length emission lacked exact result inference. I recorded the prerequisite before adding exact STRING-to-INT inference and matching binding-aware emission. I preserved the existing fixture and froze the full final integration before the passing gates above.

I do not claim bootstrap, Darwin acceptance or whole-target C99/C11 conformance. Parent6ade remains open for concat ownership/order, expression blocks, match typeof, option/capture behavior and documentation alignment. Canonical merge precedes task closure.
