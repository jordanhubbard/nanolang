# Main through PR #334 on Linux

I reconcile main `ded3ee5a` with integration `5cf21fc1` on Ubuntu ARM64.
I retain the independent ownership pass in `resource_flow.c`; main's older
identifier-only pass does not replace its lexical, branch and exit checks.
I retain source-bounded emitter metadata and keep VM slot allocation separate
from live lexical bindings. Eager checker visibility flags from main hide lifted
lambda captures; I remove those flags and verify lexical rejection together
with captured array mutation.

I retain dynamic native frames, owned arrays and scalar strings, tagged
locals/globals and recursive shape facts. Main carries older snapshots of
these same subsystems. I preserve the newer implementations and existing
acceptance cases. Main's current-frame map reclamation does not account for
caller/global/aggregate roots, so I retain execution-owned map storage pending
the existing lifetime task `task_d3310bef8bd541ba9e1e267ee213eb9e`.

I retain shared path normalization in native helpers and import main's dynamic
interpreter path normalization and 600-component regression. I retain the
newer filter emitter rather than duplicate its boolean helper and array
inference. Incoming scalar-filter tests remain in the language fixture.

## Linux findings

- Fortified GCC rejects a discarded diagnostic `write` result; I consume it
  explicitly without changing the failed-shell exit status.
- GCC diagnoses potentially truncated composed test assembly. I size those
  buffers for both input bounds and headers.
- Strict GCC rejects four generated runtime statement forms as misleadingly
  indented. I brace those conditions without disabling warnings.
- A shell wrapper turns child signals into ordinary exit codes on Linux.
  I make it execute its supervisor directly. A focused probe distinguishes
  ordinary exit 134 from `SIGABRT` (-6 in Python).

## Verified checkpoint

- C build and self-hosted bootstrap smoke checks pass.
- The shared ownership/record-pattern suite passes all 33 methods after both
  self-hosted stages and VM binaries are built.
- VM code generation passes all 77 cases, including incoming lexical tests.
- The 600-component path-normalization regression passes.
- Process-capture regression passes across its native/VM and module paths.

An early direct ownership invocation preceded the VM build and reported 20
missing-executable errors. The dependency-ordered run above passes; I do not
count the premature invocation as a product regression.

Translator, native-compiler, broad quick and full-release validation remain
separate gates. This checkpoint is not a published release.

## Subsequent acceptance

The reconciled compiler passes all 24 `test-one-ir-compiler` methods and
1,723 translator checks with 1,073 shape checks. PR #335 lands during this
run. I preserve the 1,024-local runtime, import its two regressions, and report
precise count errors from the earliest module validation. The enlarged suite
is revalidated separately.

The broad quick gate exposes duplicate merged filter predicates; I retain one
predicate with both shadow assertion sets and execute both branches' scalar
fixtures from main. Its compile/run regression passes.

Linux CI exposes missing imported callback typedefs in `sdl_nanoamp`. I align
signature collection with implicit module visibility, collect transitive
foreign signatures, use those typedefs in foreign declarations and lower
opaque callback arguments as pointers. A standalone imported-callback native
compilation test and both strict SDL example regressions pass. The compiler
contract script and destination checks pass.

The parser contract fixture reveals an ARM64 varargs defect: its final enum
constant is promoted to four bytes but read as `int64_t`. The parsed node kind
is 38; the expected array value contains unrelated upper bits. I cast integer
array literal arguments to the helper ABI. The compiled parser fixture passes.

## Clean-build and callback follow-up

I make module metadata checks depend on the generation probe and compiler/VM
binaries they execute. The standalone metadata gate passes. I give the mixer
fixture its own SDL calling-convention callback typedef, supported by SDL2.
All three mixer callback tests pass.

I verify 43 lexical, ownership and record-pattern methods together, all
code-generation cases including a lifted array capture, and the environment
and typechecker suites. The translator and sanitizer runs each pass 1,730
checks plus 1,073 shape checks. The VM run passes 272,379 checks and its
allocation-failure recovery tests.

## Foreign shadows and packaged execution

I share the VM foreign binding and supervised shadow runner with the C seed.
Module graphs declaring retained foreign policies use that runner; a failing
shadow does not fall back to another backend. I preserve callback signatures
and module-local names in generated native module objects. Three portable
cases cover owner and worker execution, dependency failure preventing output,
and the machine-readable completion report. They pass on Linux ARM64 and
Darwin ARM64. All 18 shadow supervision/regression methods pass on Linux.

On puck I compile and run the dispatch counter and API lesson with the C seed,
and pass the VM captured-callback/isolation test. Linux executes the declared
unavailable branch of module shadows; this is not evidence of Linux dispatch
support. I keep every example in compilation coverage.

I add resource_flow.o to the packaged interpreter link. All five wrapper
unit cases and seven wrapper integration methods pass on Linux.

## Example and instrumentation follow-up

All 242 eligible examples compile to bytecode on Linux after I install the
required libuv, Bullet and GLUT development libraries, preserve C++ driver
language during retained-input replay, and run boids force passes sequentially
where dispatch is unavailable. I retain all six existing intentionally invalid
or non-program exclusions; I add none. Boids' one-frame shadows also execute
through the C compiler on Linux.

The cache-publication suite passes 55 methods with eight platform-specific
skips. C++ language/reuse/header-invalidation acceptance passes on Linux and
Darwin. Its Clang fixture explicitly disables the driver's deprecated .c-as-C++
compatibility warning: diagnostics deliberately withhold reuse evidence, so
warning-bearing production inputs continue to rebuild safely.

On Darwin, the sentence example finds its bundled dictionary when built from
examples/ and completes both compilation shadows and execution. I retain the
10-second shadow deadline. The coprocessor protocol passes 35 checks, and its
changed diagnostic copies compile under strict GCC with ASan/UBSan.

Scalar async declarations now retain their function identity in bytecode and
scalar await evaluates its operand synchronously. The async fixture executes
successfully. My valid-resource fixture explicitly destructures its owned
parameter and also passes executed bytecode acceptance. These checks do not
establish promise scheduling or dynamic effect handlers; effect execution is
still a separate failing release gate under active implementation.

## Coverage wrapper links

I compile the small wrapper source separately, then link it with the runtime's
required sanitizer/coverage flags. This retains the instrumentation runtime
without creating coverage output in a private directory removed before program
execution. In a separate checkout built with real gcov instrumentation, all
five wrapper unit cases and seven publication methods pass, including literal
paths, overlapping failure, preserved destinations and staging cleanup.

## Main through PR #340

I reconcile main 2711c6eb while retaining dynamic native storage and the
existing checked direct-call formatter. The incoming wide-call test is byte
for byte identical to my existing fixture. I import both string-array write
regressions, preserving alias writes and invalid-payload rejection. Recursive
shape validation rejects the invalid payload before the older runtime-kind
check, so that diagnostic assertion follows the retained implementation.
The combined translator suite passes 1,739 checks and 1,073 shape checks.

## Integrated checkpoint after main PR #340

I reconcile main through `2711c6eb` in `b21fbeed`, preserving the candidate's
dynamic native storage and importing both string-array regressions from #338.
My translator passes 1,739 checks and 1,073 shape checks on Linux ARM64.

I repair retained C++ input replay, the bundled dictionary path, strict
coprocessor diagnostics and the scalar synchronous async lowering. The module
publication suite passes 55 methods with eight platform skips. The C++ fixture
passes on Darwin with its deprecated file-extension warning explicitly
acknowledged; production warning-bearing inputs remain ineligible for reuse.

I link packaged wrappers against instrumented runtime objects using a private
source/object/link sequence. Actual coverage objects pass five wrapper unit
cases and seven publication methods, without leaving private coverage files.

My Linux `make test-quick` run begun at `2e5967e2` passes, including all 242
eligible VM examples, laboratory frontends, Forth word sets, PTY and IDE smoke.
Only documentation changed while that run executed. This is a checkpoint,
not validation of the subsequent effect implementation.

I integrate real VM effect dispatch in `d50737cd`, preserving scalar async
lowering. Its isolated implementation passes 88 codegen, 272,403 VM, 2,676
ISA, 96 verifier and 33 schema tests plus opcode coverage. With the existing
async and resource fixture repairs, all 175 selected programs verify and both
VM dispatch implementations agree on all 175 outputs and statuses. There are
no exclusions. The implementation and boundaries are recorded in
[my VM effect evidence](vm-effect-dispatch.md). Final combined native/VM tests
remain required before publication.

## Combined effect and platform checkpoint

At `4373abc5` I integrate native and VM effects, independent interpreter union
string payloads, the launcher shadow fixture and shared effect state in
packaged wrappers. I restrict capture source identity to emitted symbols;
changing the environment's whole-file lookup context had broken ordinary
module string lowering during bootstrap. My corrected clean build and
three-stage bootstrap pass on Linux ARM64.

The exact same revision passes a fresh Darwin ARM64 compiler/VM build, all
14 shared/native effect tests, 39 C environment checks, 10 lexical tests, and
all 39 snippets through the compiled executable guide checker. The wrapper
regression loads a real foreign module that references the shared effect TLS.
Darwin emits SDK text-stub linker warnings while building the guide checker;
all these commands exit successfully.

The union regression reproduces the previous heap-use-after-free and passes
with independent payload storage under ASan/UBSan and LeakSanitizer. Native
effect tests separately pass ASan/UBSan with leak detection disabled. The VM
ownership harness executes 52,000 recursive handler activations with leak
checking enabled and returns to its stack, frame, handler and collected heap
baselines. These are bounded ownership tests, not whole-language proofs.

The earlier full run at `0e55bbbe` stopped at the UI fixture's missing math
library. I fix that link and Darwin's SDL2 runtime lookup; the focused test
passes on both hosts. The first combined clean build at `d8ba6c47` exposed the
module lookup regression above. I stop the subsequent `be5ae648` checkpoint
run after integrating the final wrapper and launcher fixes, then restart the
clean full gate at `4373abc5`. I do not count that interrupted run as a pass.


## Main globals reconciliation and full-gate follow-up

At `807dc593` I merge main through PRs #346 and #350 (`dda0290e`).
I retain the candidate translator source unchanged: its tagged global values,
array identity, initialization and ownership checks cover a broader contract
than the incoming typed-global representation. I add the incoming cross-function
array identity case and test uninitialized integer-result consumption against
my tagged-value contract. The resulting translator gate passes 1,745 checks
and 1,073 shape checks.

The clean full run at `4373abc5` passed its build, bootstrap and unit prerequisites
but stopped in the unusual-source-path module compilation regression. I repair
that private staging path in `09114660` and verify 24 invocation cases plus
47 cache-publication cases with eight existing compiler/platform skips.
I also make the list failure fixture selector volatile across longjmp.

A diagnostic continuation exposed bootstrap's mutation of the installed compiler
symlink. In `38fc36fe` I pin each test entry point's intended compiler. Three
regression methods cover seven selection, override and failure scenarios. With
an explicit C-reference compiler, the corpus passes 17 language cases and 178
application cases; its unit portion stops at one misplaced imported fixture and
two live-MAC shadow deadlines. I track those repairs separately. These diagnostic
continuations are not a passing full-suite run.


At `449b8b95`, I integrate hermetic MAC query shadows and portable corpus
discovery. Linux's 30 standalone unit programs pass without an offline CLI.
Darwin's compiler-selection regressions expose unsupported Bash globstar
behavior, including a vacuous negative-suite pass. Portable NUL-delimited
traversal retains root, nested and whitespace-bearing fixtures, and execution
uses argument arrays. Selection regressions pass on Linux and Darwin; the
Darwin negative suite passes all 37 cases and the bounded unit run passes
30 with the existing offline fixture. The combined normal-PATH Darwin run
is tracked separately.

The diagnostic tail on `4373abc5` also compiles all 242 eligible VM examples
with unchanged exclusions and passes the stdlib documentation coverage check.
The real negative corpus passes 37/37 with the C seed. Cross-backend coverage
with that compiler reports seven native executions and fourteen structural
checks, no failures or skips. Those results do not imply target execution for
the structural checks or a completed full-suite gate.


The diagnostic continuation from GLUT through the end of `test-impl` exits
successfully on `4373abc5`, with an explicit C-seed selection. It covers module
cache/dependency checks, failed-import publication, VM examples, stdlib docs,
Jackson Forth evidence and graphical/PTY smoke, interpreter examples, benchmark
measurements and property-test smoke. Gforth differential execution is skipped
because Gforth is absent on this local host; CI installs it. This continuation
is a discovery aid and does not replace the clean combined release gate.


## Combined candidate and Darwin closure

At `094cfa803b96a6d61925ab25d4cf15cfe502a825`, I start a fresh checkout with
`make clean && make -j8 && make test TEST_TIMEOUT=3600`. The clean build and
bootstrap pass. The full run passes 272,403 VM checks, 1,745 translator checks,
1,073 shape checks and all 175 verifier/equivalence programs, then continues
through later acceptance. The one-hour outer test budget preserves every test.
I do not describe this still-running gate as completed here.

A fresh Darwin checkout of that exact candidate passes `make stage1`, all five
example regressions and `make examples EXAMPLES_TIMEOUT=2400`. Its original
availability selection builds 178 targets. Installing freeglut and exposing the
existing readline keg through its real pkg-config path selects 187; all nine
additional OpenGL/readline targets compile too, with no availability override.
The strict CI failure on missing GLFW/GLEW is corrected by explicit dependency
installation, not by reducing selection.

The module link closure no longer silently truncates at 2,048 bytes. Three
Linux/Darwin tests cover cold/cached long closures, generation identity and
explicit compile-flag overflow rejection. Darwin builds the SDL launcher with
the identical long cache path that reproduced missing runtime symbols.

The combined Darwin unit corpus passes 30/30 in the normal environment at
`449b8b95`, and its 29 focused methods cover module staging, compiler selection
and empty arrays on both backends. Compiler sources are unchanged from its
fresh `6bf18479` build at that checkpoint.

The hosted sanitizer build and bootstrap pass at `094cfa80` under the explicit
60-second budget. Default shadow supervision remains ten seconds. Final hosted
tests remain separate acceptance. The release-version regression checks actual
CLI output and future generated metadata; I no longer report the stale 0.2.0
version from the public C compiler.
