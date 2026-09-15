# PR #274 reconciliation

I reconciled the eight-file change in
[PR #274](https://github.com/jordanhubbard/nanolang/pull/274), head
`f8bcb64caa240ad58f7182683ad5d0a59a2483f5`, on 2026-09-15.

`interpreter_ffi.c`, its header, `test_ffi.c` and the native ABI fixture are
byte-identical to the incoming head. Checked foreign dispatch, source call-site
accounting, array-read aliases and the incoming evaluator tests are already
present. I retain later handler-return propagation and its tests. I also retain
the creator's dependency-shadow default: the older claim test's implicit
root-only expectation must not replace the explicit opt-out now tested.

I remove the duplicate libffi Make flags introduced by automatic merging. The
resulting production and test files are unchanged from the integration parent;
this merge records reviewed ancestry and evidence, not a second implementation.

Historical CI is not green. In run `34775212890`, build, coverage and sanitizer
jobs fail because `obj/test_interpreter_ffi_native.so` has no build rule. The
Darwin strict job fails to link `crypto` (and also logs a Python environment
installation error). Integration has an explicit fixture build rule and
override-aware Homebrew OpenSSL include/library flags. I checked the expanded
fixture command. The local historical failure log is
`/tmp/nanolang-pr274-ci.log`; these old failures are not passing CI evidence.

The only PR review comment reports Copilot quota exhaustion, not approval.
I reconcile under the user's authorization and retain the original head as a
merge parent. The PR stays open until integration reaches main. Original task
`task_17fe744141024df08c2ef3de7599865a` is stopped and unowned; MAC refuses claims.

These interpreter checks do not establish complete aggregate FFI, thread-safe
interpreter reentrancy, callback lifetime safety across all backends, or release
readiness. Those boundaries remain separate work.

`make test-ffi test-eval` passes after bootstrap rebuild on Darwin
(`/tmp/nanolang-pr274-gates.log`). The three selected foreign language-claim
methods do not all pass: `test_qualified_and_returned_foreign_dispatch` fails
for its successful `map` route because indexing a float map result is inferred
as int. Typechecking rejects the float equality before producing shadow JSON.
The other two methods pass. Log: `/tmp/nanolang-pr274-claims.log`.

I preserve the failing assertion and record the dependency on map result typing
task `task_75b340982b6cf797f29b38c1a188aab3`. I land ancestry and this finding on
the integration branch, not main. The roadmap acceptance item remains unchecked;
neither this merge nor passing low-level FFI tests establish the source-level
contract as complete.
