# My private mutable-array eligibility evidence

I recorded task `task_2289d9feecbc4187937fbec82c5d2ffb` and contract `d48ff4c6`
before implementation. My frozen production/test checkpoint is `24b9d134`,
based on main `9700248a` (merged packed scalar PR700).

I add only a private analysis API and its test target. No existing verifier,
LLVM/Wasm profile, runtime instruction or production source list changes.
The new target links the analysis directly against the existing NanoISA objects;
no canonical host module acquires a transitive dependency on this unused API.

I track finite tags and array origin sets, fixed packed-versus-boxed storage,
and weak boxed-content summaries. Loops, calls, recursive SCCs and global
changes iterate until stable. Repeated allocations at one site share one
conservative summary. I distinguish invalid ordinary modules, unresolved
eligibility, explicit resource limits and allocation failure. Failed analysis
leaves the caller's report pointer unchanged and releases partial state.

My report covers shape obligations only. Possible wrong receiver/index tags
are counted as still requiring runtime checks. I do not prove bounds or precise
call/initializer ordering, nor erase possible global writes or missing values.
Unknown function returns/global escapes remain unresolved. Unsupported scalar
transfers remain unresolved instead of inheriting an approximate full-ISA rule.

My static draft audit caught an incorrect interpretation of decoded absolute
jump offsets as instruction indices. I corrected the mapping before execution;
recursive helpers at nonzero code offsets exercise it. The initial focused
run then stopped on textual bool operands and an invalid-stack fixture rejected
by the checked assembler. I retain `/tmp/nanolang-array-eligibility-focused.log`,
use numeric bool operands, and check absent-input API status directly. These
fixture failures are not runtime defect evidence.

The corrected focused run passed 11 methods in 1.042 seconds. After adding
explicit unknown-escape checks, mutual recursion, all advertised limits, and
VM result-tag assertions, the final focused run passed 12 methods in 1.273
seconds (`/tmp/nanolang-array-eligibility-final-focused.log`). Each analyzer case
runs under the ordinary C compiler and Clang ASan/UBSan. Selected eligible cases
also execute in the VM; unresolved packed/nested paths are static-report tests,
not executions of raw mismatched union writes.

Coverage includes the finite packed coercion matrix, heterogeneous boxed leaf
writes, split-result mutation shape, alias/global effects, branch origin unions,
reordered call sites, repeated allocation sites, recursive/mutual calls, loop
backedges, initializer/reentry overapproximation, optional reads/pops, unknown
escapes and unsupported instructions. Every private allocation point is failed
in order, including final report publication. Code/metadata printouts and
ordinary profile results remain identical before/after analysis. Existing
LLVM/Wasm mutation refusals preserve prior output.

I do not claim mutable opcode admission, ownership adapter completion, native
mutable-array execution, or a new source bootstrap. Nested/nominal tracing,
cycles, remaining scalar transfer coverage and matched adapters remain required
under parents488/51da. Darwin sanitizer7ba and evaluator791a remain separate.

My full affected gate passed on the frozen source:

| Gate | Methods | Seconds |
| --- | ---: | ---: |
| Private eligibility | 12 | 1.282 |
| Existing profiles | 1 | 0.368 |
| Runtime packaging | 2 | 2.166 |
| String/boxed/packed core | 3 | 5.222 |
| Existing string core | 3 | 2.926 |
| Managed operations | 45 | 57.683 |

I retained `/tmp/nanolang-array-eligibility-full.log` and used
`NMS_NATIVE_CLANG_FLAGS=--gcc-install-dir=/usr/lib/gcc/aarch64-linux-gnu/13`
with `make test-managed-array-eligibility test-verifier-profiles test-llvm-managed-strings`.
The analysis/probe are instrumented in the Clang run; linked existing NanoISA
objects keep their normal build flags. This does not claim a sanitizer rebuild
of every existing parser/verifier dependency.
