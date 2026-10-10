# My File/TCP compiler-input qualification

I qualify the compiler-input batch above parent `a00bcf2ad` on Darwin arm64.
I retain the complete bootstrap manifest and a separate recheck of all 1,022
source hashes, six host-library hashes and 17 successful steps. My raw Stage1
and Stage2 modules contain 637,268 identical bytes. My installed Stage2 also
compiles and runs hello with the C seed removed. This is local evidence, not
exact-candidate Linux acceptance or proof of compiler correctness.

I retain compressed command logs beside this document. My Stage1 API and mixed
acquisition checks pass all three methods. Snapshot checks pass with Apple
Clang, GCC16 and LLVM ASan/UBSan. My C acquisition and parser ownership/refusal
checks pass; bootstrap mutation/dependency checks pass 17 methods and workflow
checks pass seven. The final combined generation suite is recorded separately.

I preserve failed attempts: duplicate descriptor symbols in a combined native
consumer, undefined symbols in the rejected shared-library-only arrangement,
and a mixed-input fixture that accidentally supplied File bytes for TCP. The
corrected module uses private descriptor names; the fixture now includes both
a valid TCP document and an explicit crossed-catalog refusal. My premature
bootstrap failed its unchanged host-closure guard and is not acceptance
evidence. The final frozen bootstrap passed that guard.

I also retain the original hosted sanitizer job log. Its Stage1 self-compile
timed out after 1,800 seconds. The actual VM compile command already used
`-O3`; omission of optimization in a later command does not explain that
runtime failure. I change the workflow to run real `bootstrap3` before unit
tests and keep compile/link flags consistent. The runtime cause and a passing
hosted result remain open under #982.

I admit checked File/TCP companion acquisition and immutable catalog queries.
I do not yet admit TCP namespace/type/lowering, wire metadata, service dispatch
or selected network shadows. Full Socket/Conn execution, public connect and
WebSocket integration remain open under #990 and the full 5.1 contract.

My first final eight-method batch passed the Stage2 API/mixed acquisition,
C plan ownership and C/full-frontend refusal checks. It failed two gates:
the full descriptive corpus reached its last shadow then exceeded10 seconds,
and the actual-driver test expected a pre-implementation File diagnostic.
I preserve that batch log. I update the diagnostic to the actual missing-grant
refusal and give only this58-case, exact1MiB stress corpus30 seconds, retaining
all selected shadows and boundary assertions. I record the corrected run
separately; the ordinary compiler default remains10 seconds.

My corrected paired corpus and actual-driver protection gates pass both methods
in89.724 seconds. C seed, Stage1 and Stage2 select all expected shadows and
produce identical complete catalog/corpus output. I retain their command,
terminal and shadow-selection records. The separate C ownership/budget check
also passes with LLVM ASan/UBSan. These corrections change tests only; my
bootstrap source and host hashes remain unchanged.
