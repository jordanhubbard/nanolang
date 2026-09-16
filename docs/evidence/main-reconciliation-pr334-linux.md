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
