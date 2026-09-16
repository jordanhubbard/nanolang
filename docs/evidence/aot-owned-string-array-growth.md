# I grow string arrays without a fixed append arena

My native compiler exhausted a 65,536-pointer arena while appending the 331st
string to an output array. Each append copied all preceding elements into new
arena space. Increasing that arena would preserve quadratic accumulated space
and another arbitrary limit.

I now grow an owned pointer buffer geometrically behind the same array handle.
Aliases observe the updated length and data. I check element-count overflow,
byte-size overflow, allocation failure and inconsistent owned storage before
writing. I retain the old allocation if `realloc` fails, then abort; I do not
claim recoverable allocation errors.

My internal AOT string-array descriptor records its owner. A borrowed descriptor
is copied into owned backing storage on its first growth. Cleanup never frees
the borrowed descriptor or its original buffer. This changes my generated-C
internal representation, not the foreign `DynArray` ABI.

I track owned buffers and handles until generated `main` completes. Foreign
walk-result strings are copied into a separate owned list and remain alive even
if their array slot is replaced. I free both lists after entry returns, after
record/map cleanup. Ordinary literal and borrowed strings are not individually
freed. This is execution-lifetime ownership, not per-object garbage collection
or a bounded-memory guarantee for an indefinitely running program.

## I test growth, aliasing and cleanup

My regression executes 70,000 appends with fewer than 26 allocation calls,
checks values through an alias after reallocations, constructs a 70,000-element
literal, grows a borrowed stack-backed array without freeing its input, and
checks an escaped copied string after its slot is replaced. Instrumentation
checks that cleanup leaves zero owned allocations and tolerates a second cleanup.
The generated entry then executes and cleans up another array.

Twelve negative runs cover allocation failures for handles, owner metadata,
growth, borrowed-buffer adoption and foreign-string copies, plus length overflow,
byte-count overflow and inconsistent storage. All must trap. The real standard-
library adapter regression also remains in compiler acceptance.

I retain helpers for unused foreign-array adapters without loading their
libraries. The existing unused-adapter test caught this emission dependency.

`make test-nvm2c` and `make test-nvm2c-sanitizers` each pass 1,670 AOT and
1,073 shape checks. The latter verifies fresh ASan/UBSan translator and shape
objects; leak detection is disabled, so my allocation-count regression provides
the separate cleanup evidence. `make test-one-ir-compiler` passes 19 of 20
methods; the full compiler method remains failed. `git diff --check` passes.

## I keep the next failure visible

With array growth fixed, my native compiler proceeds to `cg_build` and aborts
in `nstr_concat` while emitting its C runtime. The scalar-string arena is also
fixed-size. I track checked owned string allocation in MAC
`task_b0c4ad8c9a824e64ab8fd3fa6881146e`; I do not enlarge that arena here or report
compiler acceptance as passed.

MAC `task_7b8691dd087e48ea9afdeb89ddcf4640` tracks this array fix. Worker claims
remain unavailable, so evidence does not imply ledger closure or release.
