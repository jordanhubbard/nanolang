# I measure the remaining shadow cost

My exact integrated source `d39a976fdceeb520d610881da6f8a323f4dbfbe3`
passes fresh compiler construction and reaches 435 of 604 original shadows,
then stops at the original 60-second deadline. The loader repair removes the
earlier fork refusal. My record-name allocation change does not establish a
deadline repair. I retain that terminal under task_96b40bc1ccd644bca2131878fa6cf372.

Before selecting another correction, I sample the executing shadow child with
Linux perf at 99 Hz. The compiler remains the ordinary user's process; elevated
access is limited to the kernel profiler attached to that exact child. I verify
its parent, executable and user identity before attachment. I keep the original
input, all shadow bodies and selection, instrumentation and 60-second deadline.
The original outer compiler alarm is unchanged, with a stricter 180-second
supervisor and a separately bounded profiler. No known unsafe memory case is
introduced or replayed.

I reuse only the exact retained d39 compiler/provider bytes after hashing their
complete recorded closure. I retain source/tool maps, command arguments, raw
compiler streams, perf data, profiler diagnostics and terminal process cleanup.
Sampling overhead makes this diagnostic unsuitable for timing acceptance. I
use its sampled call stacks to choose a source change; a subsequent fresh,
uninstrumented measurement must still pass the original release deadlines.

The bounded sample completed with no lost samples, no outer timeout, no input
drift and no surviving process. My compiler again stopped after 435 shadows at
the original 60-second deadline. AddressSanitizer fake-stack allocation consumed
90.42% of sampled user cycles: 41.15% in `__asan_stack_malloc_2`, 30.57% in
`__asan_stack_malloc_0`, 15.87% in `__asan_stack_malloc_1` and 2.83% in
`__asan_stack_malloc_3`. The visible call paths include `clone_value_at` and
`record_result_publish` through `eval_preserve_value` and
`eval_staged_argument`. This identifies a measured cost; it does not establish
that removing sanitizer coverage is acceptable.

I will preserve that coverage and remove needless recursive leaf calls instead.
My aggregate snapshot copier currently calls `clone_value_at` for every field,
including scalars and borrowed reference leaves for which it only copies the
value and clears control-flow flags. I will perform that leaf operation inline
and recurse only for owned strings, records and tuples. I retain the depth check
for every child, including direct leaves, so the existing 128-level refusal does
not move. All allocation failure cleanup remains unchanged. Focused record and
tuple controls must verify that nested leaf values retain their payloads and
lose return, break, continue and return-target state exactly as before. Fresh
full-selection qualification under the original instrumentation and 60-second
deadline decides acceptance.
