# My full driver shadow timing checkpoint

I retain the b33 Linux original bootstrap and separate verbose diagnostic
terminals under task_6ec8b514c61449d0863401584e86376b. Both reached the unchanged
ten-second shadow deadline. Buffered verbose output ends near
nisa_block_falls_through; it does not identify the active local cost.

I reuse the reviewed SDK observer from7bd1c08d2 without semantic SDK changes:
the exact shadow_timing.h, main fork/callback/selection/wait phase records and
evaluator interpreter/module/shadow records. The source/item/ordinal values
follow existing traversal. Monotonic time and process CPU fields retain validity
flags; parent cumulative reaped-child CPU is not per-PID. Bounded unbuffered
writes preserve errno. The opt-in is NANO_SHADOW_TIMING=1; disabled calls do not
change acceptance. I add only owning main/eval header prerequisites here.

I compile fresh observer main/eval objects and link against independently copied,
verified unchanged b33 providers in a separate diagnostic tree. The original
full src_nano/nanoc_v06.nano input and all imported shadows remain selected.
The child deadline stays ten seconds, the diagnostic outer bound120 seconds.
I retain exact compiler/link commands, source/tool/provider/product maps and
all observer output. I execute no produced compiler as acceptance in this run.
Observer I/O and timing calls add measured work; this is not an uninstrumented
qualification result or permission to remove checks. Source review precedes
that single fresh diagnostic invocation.
