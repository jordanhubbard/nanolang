# My native function-field checkpoint

I preserve function-valued aggregate fields as distinct native function storage,
with a checked function tag and module-local target index. My extraction checks
both field storage and payload tags; dispatch checks the call site's target set.
I do not treat a function ID as a heap edge.

My 24-method callable suite passes, including the original native returned-call
methods and VM/native parity controls. Two new methods check nested records,
record arrays, record arguments/results, adjacent owned strings across at least
two observed collections, and malformed field storage/payload tags and targets.
Generated C compiles with strict C11 warnings and ASan/UBSan, then runs with leak
detection enabled through Homebrew LLVM. My first collection-count assertion
fails with 2,000 allocations per churn loop; I retain that terminal and increase
the workload to 8,000 while retaining the two-collection assertion.

My broader native gate passes 2,431 structured-C checks, 2,535 shape checks and
204 callable-analysis checks. The manifest binds these terminals to source hashes;
this working-tree checkpoint is based on the recorded commit.

My unchanged combined source-container fixture now advances beyond record fields
and refuses its function-array literal. My self-hosted producer still refuses its
record type. These remaining producer/storage paths, captured closures and full
release qualification remain open. I have not released 5.1.
