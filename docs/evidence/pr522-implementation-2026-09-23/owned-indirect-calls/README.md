# Owned callback invocation and local storage

I connect conservative callback targets to exact consuming ownership checks,
VM execution, generated native dispatch and source lowering. My
[contract](../../../NANOISA_OWNED_INDIRECT_CALLS.md) states the bounded profile.
The original generic resource-bearing callback refusals remain required.

## Backend qualification

My complete affine gate passes 6,098 checks plus 6,619 allocation/visit-failure
checks, ordinarily and with ASan/UBSan and leak detection. Its 102 target
fixtures cover 17 cases in six declaration orders: 60 accepted and 42 refused.
Accepted target plans now also require executable ownership verification.
Additional negatives keep same-shaped but nominally distinct resource
parameters and results refused even when target planning succeeds.

My 66 runtime cases cover ten admitted callback shapes in six declaration
orders and six traps before/after resource consumption or pending returns.
They include returned functions, aliases, loop/branch targets, resource
factories, consumers and void consumers. Four VM entry APIs, each reused
twice, pass 8,246 checks including activation generation, frame/stack reset,
reference cleanup and heap counts. Generated C matches CLI retranslation,
runs under ASan/UBSan with leak detection, and checks twenty repeated calls,
wrong-target/wrong-tag dispatch and allocation failures until success.

- [Ordinary affine gate](affine-ordinary.log.gz).
- [Private instrumented affine and runtime gates](owned-sanitized.log.gz).
- [Existing twelve value graphs and preflight/invocation/reuse gates](value-graphs.log.gz).

The private run instruments project VM, analyzer and emitter objects. External
libraries remain ordinary. The Python harness's CLI parity commands use the
ordinary root binaries; its generated native cleanup binaries are instrumented.
I do not claim those CLI commands are instrumented.

The first private invocation failed because the harness treated `CC` with
flags as one executable path. I parse it into arguments and retain both the
[failed run](compiler-command-before.log.gz) and the [corrected 60-case run](compiler-command-corrected.log.gz).
The later 66-case qualification includes void consumers.

## Compiler storage repair

Fresh bootstrap exposed a separate string storage conflict in
`Parser.lets[].var_type`. My [original terminal](bootstrap-shape-before.log.gz),
[verified compiler bytecode](compiler-before.nvm.gz) and [projection trace](compiler-shape-path.log.gz)
retain the failure. The bytecode's foreign imports name the original local
build artifacts; the standalone regression has no such dependency.

An unresolved projected field was equated with a local at `STORE_LOCAL`.
Repeated calls later required tagged local storage, imposing that requirement
back on the producer field. Returning its enclosing container then conflicted
with a later exact-string destination. I now use directed storage conversion
for unresolved local assignments, as for known scalar assignments. I do not
change shape-solver constraints or admit optional record fields into exact
string storage.

The [small regression](local-storage.nasm) reproduces the same rejection in
six declaration orders before the repair. All six pass VM and native execution
afterward ([before/after terminal](local-storage-regression.log.gz)). The
regression lives in the complete translator gate alongside the unchanged
wrong/absent-tag and incompatible-payload controls.

Both complete translator gates pass 2,458 assertions:
[ordinary](translator-ordinary.log.gz) and [fresh ASan/UBSan](translator-sanitized.log.gz).
The sanitizer driver uses private object, binary and File archive paths and
verifies instrumentation in translator and shape objects.

An earlier bootstrap also exposed two existing shadow assertions whose
nonfunction-call diagnostic I had changed. I restored that diagnostic and
retained the [failure](bootstrap-shadow-before.log.gz); I did not weaken the
assertions.

## Fresh source qualification

Fresh `make -j1 bootstrap` passes both native stages, their smoke checks and
the installed compiler check ([terminal](bootstrap-corrected.log.gz)). Native
binaries differ; canonical bytecode fixed points remain a separate gate.

With Homebrew LLVM, `NANO_CFLAGS='-O1 -fsanitize=address,undefined
-fno-sanitize-recover=all -fno-omit-frame-pointer'` and
`ASAN_OPTIONS=detect_leaks=1:halt_on_error=1`, I pass:

- [42 methods](source-sanitized.log.gz): all 15 resource-boundary and 18 original
  generic-function methods across C seed, Stage1 and Stage2, plus nine purity
  and returned-call methods using Stage2 (and the purity suite's C controls).
- [Nine adjacent methods with Stage1](source-stage1-adjacent.log.gz), with the
  same generated-program instrumentation.
- The [owning C transpiler gate](transpiler.log.gz).

Compiler executables and external libraries remain ordinary in these source
runs. The resource suite retains all thirteen original methods and adds
returned callback aliases and a void consumer. The original fixed resource
parameter/result acceptance failures are repaired in both fresh native stages.

## Hosted map harness repair

Both map harnesses hardcoded `cc`, ignoring the configured sanitizer compiler.
I reproduce the hosted Darwin failure locally: all eleven failing subcases
report unsupported leak detection ([before](maps-local-before.log.gz)). I now
honor `NANOLANG_GUARD_SAN_CC`, falling back to `CC` and then `cc`, and split
compiler arguments with `shlex`. I retain every sanitizer flag and leak check.

`NANOLANG_GUARD_SAN_CC='/opt/homebrew/opt/llvm/bin/clang
-fno-omit-frame-pointer' make -j1 test-map-declared-tags` passes all seven
methods ([corrected terminal](maps-corrected.log.gz)). This exercises a compiler
command containing an argument. Only generated native programs are instrumented
by these harnesses; their root CLI tools remain ordinary. A fresh hosted run
must still qualify the platform.

## Remaining hosted evidence

Hosted run `35950785135` still fails the unchanged Linux 60-second compiler
shadow deadline ([units-00 terminal](hosted-units00-before.log.gz)). Its Darwin
map gate reports unsupported leak detection with a hardcoded compiler
([terminal](hosted-macos-before.log.gz), `task_7474e0a5cc534774bb2db30fbe039299`).
The Linux timeout remains unresolved; the map harness correction needs a fresh hosted run. They do not qualify the final candidate,
and neither a pending job nor this local checkpoint makes #522 mergeable.
