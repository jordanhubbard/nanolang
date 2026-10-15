# My C File source lowering batch

I continue [#989](https://github.com/jordanhubbard/nanolang/issues/989)
from `665fcf9cb` on `release/5.1-completion-20261007`. I keep the complete
[5.1 scope](../../RELEASE_5.1_SCOPE.md) and
[File source contract](../../NANOISA_FILE_SOURCE_IMPLEMENTATION.md) open.

I now lower the actual C parser's checked File graph into service-bearing
NanoISA. I retain every function and shadow, including unselected bodies. I
assign lexical slots, direct function indices, original service imports and
all eight nominal layouts; preserve signatures, ownership declarations and
unit payloads; and emit Result branches, moves, scoped exclusive references,
scalar fields, control flow and loops. Before serialization I analyze the
complete module with the cyclic File query and retain its actual stack peaks.
The public File consumers separately validate and execute the resulting bytes.

My seven-method [source gate](source-gate.log) passes on Darwin in 64.032
seconds. It executes all five unchanged generated shadow bodies through both
VM and native C; checks a helper loop with write/rewind/read/close, returned
Result values, field projection and re-exported helpers; and checks logical
short-circuit effects and terminal call/operator operands. It also covers
assertion cleanup, explicit grant refusal, open-descriptor counts before and
after execution, selected allocation-failure prefixes, local/function limits,
unsupported shared-reference calls and preserved prior output. The generated
binding fixture is byte-identical to publisher output in my paired ownership
gate. Each selected-shadow module contains all five shadow bodies plus main
and its scalar entry wrapper.

The source gate instruments the lowerer with LLVM ASan/UBSan and leak checking.
Its native programs use LLVM `-O0` with ASan/UBSan. Common compiler objects and
the native runtime archive in that gate are ordinary builds. I additionally
compile the File query and runtime carrier directly with sanitizers for the
selected runtime controls; this distinguishes their instrumentation from the
full native archive. The final allocation-control correction reports status 4
for an output-buffer allocation failure and checks variant accessor success.

My [File flow/body gates](flow-body-gates.log) pass: 3,405 instrumented logical
flow checks, 3,266 linked flow checks, 9,507 instrumented acyclic body checks and
3,662 linked body checks. I explicitly reject clearing a live owner or formal
reference with the scalar clear spelling. The matched source VM/native loop
and branch cases exercise successful copy-local clearing.

## Failures I retain

- [Both old ownership probes](short-circuit-c-first.log) accepted a move in a
  skipped logical operand; the [Nano probe](short-circuit-nano-first.log) also
  reports status zero. Both corrected passes join the skipped and evaluated
  paths. The concrete [source](short-circuit.nano) stays in this evidence.
- My first lowerer left an earlier argument on the stack when a later Result
  expression returned. The real consumer refused the
  [module](terminal-operand-first.log). I retain the [source](terminal.nano),
  stage operands when a later expression can return, and drain pending owners
  on that exit. The final source matrix covers scalar and File arguments and
  an eager operator.
- The operator case then exposed [terminal nominal typing](terminal-type-first.log):
  a match whose arms both return has no value to compare with the enclosing
  operand requirement. Both checkers now retain that fact, and eager operators
  propagate guaranteed return. A conditional logical right operand does not
  become an unconditional return.
- The [initial source gate](initial-gate.log) includes two distinct problems:
  my test used the reserved word `byte`, and five LLVM `-O1` native builds hit
  the 120-second command deadline. I corrected the fixture to `octet` and used
  the actual FileError field `status`. About 5.6 MB of generated native C makes
  optimized compilation a separate unresolved product cost. I added process
  group supervision to the test harness. Passing `-O0` execution does not close
  optimized publication qualification.

## Reproduction and remaining work

```sh
NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang \
  make -f Makefile.gnu CC=/opt/homebrew/opt/llvm/bin/clang test-service-lowering-sanitize

NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang \
NANO_SERVICE_OWNERSHIP_DRIVER_MODULE=/private/tmp/nanolang-service-lowering-driver-final.nvm \
NANO_SERVICE_BODY_DRIVER_MODULE=/private/tmp/nanolang-service-lowering-driver-final.nvm \
  python3 -m unittest -v tests.test_service_ownership tests.test_service_bodies
```

The paired gate uses the [C-produced development driver](driver-build.log)
built from this batch's `src_nano/nanoc_v06.nano`. That component build is not a
clean Stage1/Stage2 release bootstrap. My [paired checker gate](paired-checkers.log) passes all nine methods in
327.880 seconds, including twelve valid ownership cases, eighteen invalid
cases, actual generated binding output, imported helpers and nominal refusals.
Both producers run the Nano probes in VM and sanitized native execution. The
actual drivers retain their prior-output refusal checks.

My final [instrumented runtime controls](runtime-instrumented.log) pass with
[the lowerer, File query and runtime carrier instrumented together](runtime-instrumented-build.log).
The generated read shadow returns zero, the helper loop returns 10, and the
terminal-argument source returns 7. These runs include the final output-buffer
allocation-status and output-preservation assertion. The earlier
[allocation-control run](final-allocation-control.log) also passes.

The C lowerer is currently exercised through its internal API. Actual compiler
CLI publication still stops at the service guard. Independent Nano byte
lowering, shared/multi-reference and indirect source calls, multiple nominal
catalog declarations, richer ordinary source combinations, driver grants,
supervised shadow selection and staged publication remain required. Socket,
backend coverage, exact-candidate bootstrap/platform gates, the tag and release
publication also remain open. I preserve the user's untracked guide fixture.
