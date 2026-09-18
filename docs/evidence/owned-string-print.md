# My bounded owned string and print evidence

I implemented `task_badd6be9c31a6e2eac810b95913b4f84` on branch
`feat/owned-string-print-runtime`. My production-only checkpoint is
`ae4b064c4455e083a7d6fc596392811a91bc3bae`; my tested checkpoint is
`b5861147d662a73fb00789dbc1f4f6287f315b1c`. Both are pushed. I leave the
roadmap item open until the production change receives independent review.

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
initializers keep strict Clang's missing-field diagnostics enabled.

## Focused qualification

At the tested checkpoint, `make -j8 CC=/opt/homebrew/opt/llvm/bin/clang
test-owned-string-print` passes with Homebrew Clang 23.1.1. The generated
native programs run with ASan and UBSan and explicitly select
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
  graphs.

The log is
`/private/tmp/nanolang-owned-string-print-llvm23-lsan-b5861147.log`, SHA-256
`73fdd1d02ee814c91d6227f3dca3c2e0c6d502013b8c527e99e880d38b91bdef`.

## Adjacent ownership qualification

The unchanged owned-value graph, owned/void result, single consuming-call and
multiple consuming-call suites all pass under the same explicit Homebrew LLVM
selection. This includes their graph bounds, recursion, ownership authority,
allocation, preflight, verification-reuse and generated native sanitizer
checks. The log is
`/private/tmp/nanolang-owned-string-print-adjacent-llvm23-b5861147.log`,
SHA-256
`f8180d879019837c7a029b25001dd939a6d5e10667a72521f8eb592ca15fd267`.

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
