# My reference evaluator process contract

I track `task_3d5bf23a0b40466aa8ba4e7c84e31e01`. I run nanocore-ref with direct
argv and explicit stdin/stdout pipes. I preserve the adjacent-compiler lookup,
then PATH fallback, without a fixed-size path or shell-command buffer. I send
the expression's exact bytes followed by one newline. Returned output remains
heap-owned and has trailing CR/LF removed; empty output or process/transport
failure returns NULL. Nonzero or signaled child exit is an error even if it
printed text. Stderr remains suppressed.

A small writer child performs only close/write/_exit while the parent drains
stdout. This permits output before all input is consumed without pipe-capacity
deadlock and confines SIGPIPE to the writer. An inherited ignored SIGPIPE still
produces a checked write error. I close unused owned pipe ends in each process,
use close-on-exec descriptors, and reap both children. On a local transport or
allocation failure I kill and reap only my owned children. I do not introduce a
new execution deadline for the trusted evaluator in this bounded child.

I do not use or change the exporter SBuf implementation. My tests compile this
corrected source and invoke a benign evaluator stub with ordinary quoted text,
spaced paths, large input/output, early exits and repeated calls. I check error
results, descriptor and child cleanup. These tests establish process transport,
not formal semantic correspondence or the reference evaluator's completeness.
