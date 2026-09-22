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
