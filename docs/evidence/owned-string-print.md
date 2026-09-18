# My bounded owned string and print evidence

I implemented `task_badd6be9c31a6e2eac810b95913b4f84` on branch
`feat/owned-string-print-runtime`. After rebasing without conflict onto
canonical main `44ad5f69`, my review-correction checkpoint is
`f7a66804f718ab34983e2aa94a433a4b6c36a31f`. The earlier `ae4b064c`
production snapshot is superseded: it retained two positional initializers
from the two-field carrier. The corrected checkpoint has no positional
`nown_value` initializer. I leave the roadmap item open until the production
change receives independent review.

I admit only mode-zero `TAG_STRING` parameters and their exact locals in my
bounded owned value-call graph. My record-field guard still accepts only INT,
BOOL, U8 and STRUCT. I accept validated, NUL-free module literals and consume
them only through `PRINT` or `PRINTLN`. I still reject string results, resource
fields and broader string operations.

My VM preflights every instantiated literal before it creates an owned
activation. Each public PRINT trap invalidates the invocation proof, releases
the popped trap root exactly once and resumes through checked admission. The
proof fixture observes 16 verifier admissions across one four-print invocation;
this is evidence that the proof was not retained across host output traps, not
a stable performance count.

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

The focused gate covers:

- nonempty, empty, repeated and sibling literal output through three helper
  frames, with exact captured bytes and newline placement;
- output followed by success or assertion failure through `vm_invoke`,
  `vm_execute`, `vm_call_function` and `vm_invoke_callable`;
- VM constant-table, stack, frame, reference and trap cleanup across repeated
  execution;
- separately injected VM setup and owned-record allocation failures, followed
  by successful reuse of the same module and, for frame failure, the same VM;
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
- refusal before activation when a missing instantiated literal reaches the
  traced fallback admission configuration.

The review log is
`/private/tmp/nanolang-owned-string-print-review-f7a66804.log`, SHA-256
`f76b374e603f09b84064caa5acd7a57540ba37f58c5ddf3044ee7b955cd9c79f`.

The readiness correction was qualified from a clean tree at `1c986ec8` with
the same explicit Homebrew Clang 23.1.1 and LeakSanitizer selection. Its focused
log is `/private/tmp/nanolang-owned-string-print-readiness-1c986ec8.log`,
SHA-256
`4d57c0346780d30ce2960a35ac35142b455ff7cb78422fe0f220c4325de8fc8c`.

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
