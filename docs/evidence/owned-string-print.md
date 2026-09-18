# My bounded owned string and print evidence

I implemented `task_badd6be9c31a6e2eac810b95913b4f84` on branch
`feat/owned-string-print-runtime`. After rebasing without conflict onto
canonical main `44ad5f69`, my review-correction checkpoint is
`f7a66804f718ab34983e2aa94a433a4b6c36a31f`. The earlier `ae4b064c`
production snapshot is superseded: it retained two positional initializers
from the two-field carrier. The corrected checkpoint has no positional
`nown_value` initializer. I leave the roadmap item open until the production
change receives independent review.

After rebasing onto canonical main `96cdb7c5`, my admission-boundary
checkpoint is `8a87f9225e022a8953dc08bbf8a2125013484ae8`. It retains the runtime
corrections above and makes the review claims independently observable: each
output resume has exact admission accounting, incomplete constants are tested
through fast, trace, callback, reference and direct-core paths, and borrowed
`PRINT`/`PRINTLN` refusals do not depend on an earlier `PUSH_STR` refusal.

The independent hold identified that those checks were still caller-side and
late. At production checkpoint
`d2aade9b57116be2f4344ca24cf21776be5eb309`,
`vm_ownership_supported` itself requires constant readiness before returning
true. The shared affine and runtime-verifier opcode allowlists now require the
value-graph classification for `PUSH_STR`, `PRINT` and `PRINTLN`; their later
analysis and native-emitter refusals remain in place.

I admit only mode-zero `TAG_STRING` parameters and their exact locals in my
bounded owned value-call graph. My record-field guard still accepts only INT,
BOOL, U8 and STRUCT. I accept validated, NUL-free module literals and consume
them only through `PRINT` or `PRINTLN`. I still reject string results, resource
fields and broader string operations.

My VM preflights every instantiated literal before it creates an owned
activation. The first public PRINT trap removes the invocation proof; later
PRINT boundaries retain that proofless state. Every boundary releases its
popped trap root exactly once and resumes through checked fallback admission.
The proof fixture uses a one-function four-print sequence with no helper
activations. It checks exactly three verifier admissions on each of the four
resumes and 13 total including initial admission; these counts describe the
current checked path rather than a performance contract.

My specialized native carrier now has an immutable pointer and exact length.
It writes those bytes with `fwrite`, writes one explicit newline for
`PRINTLN`, and never treats a string view as an owned record. Designated scalar
initializers keep strict Clang's missing-field diagnostics enabled. Record
allocation writes `.record` only into a zero-initialized scratch carrier and
resets that carrier with `{0}` after transfer.

I also keep all three string opcodes inside the owned value-call profile. A
borrowed `CALL_REF` graph cannot gain string literals or output through the
shared affine analyzer or the native emitter. VM admission checks instantiated
constants before choosing either the invocation-proof path or its conservative
fallback; the traced fallback refusal is exercised directly.

The follow-up `task_a468c371da11fb8a37a9567bb7f0af21` closes the public-core
readiness gap at rebased checkpoint
`1c986ec8efbeb9930e74a8cc537b0396de90d1fd`. I inspect the active module's
actual constant table before every owned core entry, including direct public
entry and a resume after a host-output trap invalidates the invocation proof.
An incomplete table refuses before another instruction executes and clears the
reference activation through the existing error path. A fresh complete-table
activation still crosses PRINT and ASSERT traps and returns normally. An
ordinary module with no ownership metadata still takes the checked opcode path;
its missing literal produces the existing decode refusal rather than being
reclassified as owned execution.

## Focused qualification

At the review-correction checkpoint, a clean build followed by `make -j8
CC=/opt/homebrew/opt/llvm/bin/clang test-owned-string-print
test-caller-reference-analysis` passes with Homebrew Clang 23.1.1. The
generated native programs compile with `-Wall -Wextra -Werror`, run with ASan
and UBSan, and explicitly select
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`.

The focused gate at `8a87f922` covers:

- nonempty, empty, repeated and sibling literal output through three helper
  frames, with exact captured bytes and newline placement;
- output followed by success or assertion failure through `vm_invoke`,
  `vm_execute`, `vm_call_function` and `vm_invoke_callable`;
- VM constant-table, stack, frame, reference and trap cleanup across repeated
  execution;
- reached `heap.c` setup failures for the intern bucket and module-string
  objects (10 allocation attempts in this fixture), followed by fresh-VM
  recovery, and both owned-record allocation attempts during invocation,
  followed by successful reuse of the same VM;
- byte-identical native regeneration, generated native allocation failure and
  plain native entry execution;
- canonical text round-trip of literals, parameter tags and ownership
  descriptors; and
- refusal of embedded NUL, absent or out-of-range literals, wrong parameter
  tag or mode, string results and resource fields, unsupported string stack
  operations, missing or extra PRINT operands, tail calls and recursive value
  graphs;
- direct refusal of `PUSH_STR`, `PRINT` and `PRINTLN` outside the value-call
  profile, including a borrowed `CALL_REF` fixture; and
- refusal before activation when a missing instantiated literal reaches fast,
  traced, callback, reference or direct public-core admission; and
- exact proof invalidation and three fallback verifier admissions at each of
  four output boundaries in a helper-free sequence, with 13 admissions total.

The review log is
`/private/tmp/nanolang-owned-string-print-review-f7a66804.log`, SHA-256
`f76b374e603f09b84064caa5acd7a57540ba37f58c5ddf3044ee7b955cd9c79f`.

The readiness correction was qualified from a clean tree at `1c986ec8` with
the same explicit Homebrew Clang 23.1.1 and LeakSanitizer selection. Its focused
log is `/private/tmp/nanolang-owned-string-print-readiness-1c986ec8.log`,
SHA-256
`4d57c0346780d30ce2960a35ac35142b455ff7cb78422fe0f220c4325de8fc8c`.

The final boundary qualification was clean at `8a87f922` with Homebrew Clang
23.1.1 and `detect_leaks=1`. It passes 114 caller-origin analysis checks, 600
allocation checks with 10 setup and 2 invocation attempts, 62 proof checks
with 13 admissions, and the VM/native output and cleanup method. Its log is
`/private/tmp/nanolang-owned-string-print-boundaries-8a87f922.log`, SHA-256
`0e33652e2e4891dc0ee792f326c4dabc4fe2daad29a459cd786c33aa268ad99a`.

The positive-predicate qualification was clean at `d2aade9b` with Homebrew
Clang 23.1.1 and `detect_leaks=1`. It directly checks that missing constants
make runtime readiness, `vm_ownership_supported`, fast/fallback admission and
public core all refuse without invoking the owned verifier in each of the
fast, trace, callback and active-reference configurations. The borrowed
profile separately refuses `PUSH_STR`, `PRINT` and `PRINTLN` in affine
analysis, the runtime verifier and native emission. The gate passes 117
caller-origin checks, 600 allocation checks and 130 proof/readiness checks;
the exact four-output sequence still records 13 admissions. The log is
`/private/tmp/nanolang-owned-string-positive-paths-d2aade9b.log`, SHA-256
`36dbad741e4bbb202c254960f4b54292bf2c5cccaebec58762e2c881a81420bd`.

## Adjacent ownership qualification

The unchanged caller-reference, owned-value graph, owned/void result, single
consuming-call and multiple consuming-call suites all pass under the same
explicit Homebrew LLVM selection. This includes their graph bounds, recursion,
ownership authority, allocation, preflight, verification-reuse and generated
native sanitizer checks. The log is
`/private/tmp/nanolang-owned-string-print-adjacent-f7a66804.log`, SHA-256
`53ef47cb847f6ae7da76a6f0434291b0d2a72c3dc257f8c5eb2ff5616c68c160`.

I reran the same adjacent ownership suites after the readiness correction.
They pass at `1c986ec8`; the log is
`/private/tmp/nanolang-owned-string-print-readiness-adjacent-1c986ec8.log`,
SHA-256
`c7ddf8539fe281e3ed55aae0f26f5fe8758c063ef90feba9f3840e504a7d3ec2`.

I reran those suites after the exact boundary corrections at `8a87f922`.
They pass with the same Homebrew Clang 23.1.1 selection. The log is
`/private/tmp/nanolang-owned-string-print-adjacent-8a87f922.log`, SHA-256
`b9c19568ed68ac4bc6d680df2936b91047741ef3fd8b62049bca5da2c54a3abe`.

I reran them after the positive predicates and shared allowlists were closed at
`d2aade9b`; they pass unchanged. The log is
`/private/tmp/nanolang-owned-string-positive-paths-adjacent-d2aade9b.log`,
SHA-256
`b22b605b749419e91039f4cd685683dc3b863f07f178e71d621255265731f2d2`.

I preserve the first default-compiler adjacent run separately. Its ordinary C
and VM fixtures pass, then its older Python harnesses request LeakSanitizer
from Apple Clang and stop with `detect_leaks is not supported on this
platform`. I did not weaken those tests. That log is
`/private/tmp/nanolang-owned-string-print-adjacent-b5861147.log`, SHA-256
`1070bca1e08384080ab8cd701ae487e4493b313cc9ef75b5d7c4da1024c88579`.
It is not a substitute for the passing Homebrew LLVM gate.

I do not claim source-producer admission, acceptance of the unchanged affine
example, product PR #522, general string semantics, other native backends or
release readiness from this bounded runtime evidence.
