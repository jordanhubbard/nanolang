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

## My measured acceptance

At source `3237b994` on main `38f29203`, seven methods pass with strict O2
GCC ASan/UBSan/LSan (1.156 seconds) and Clang ASan/UBSan/LSan (1.076 seconds).
I retain `/tmp/nanolang-reference-transport-gcc-final.log` and
`/tmp/nanolang-reference-transport-clang.log`; the initial six-method GCC gate
also passed before I added the allocation-refusal control.

My benign stub receives ordinary quoted text literally from an adjacent path
containing spaces; a separate spaced PATH entry exercises fallback lookup.
I transfer an expression exceeding 600,000 bytes while the child first emits
262,144 bytes, checking complete output beyond pipe and old fixed-buffer
sizes. I check CR/LF trimming, empty output, nonzero exit, missing program,
early input closure with default and ignored SIGPIPE, thirty repeated calls,
and initially closed standard descriptors. A test-compiled realloc shim
refuses output growth after child startup; the API returns NULL and reaps both
children without descriptor or parent allocation leaks.

The test harness checks that no owned child remains waitable after every call
and compares its ordinary descriptor count before and after. No shell payload,
known-aborting compiler, or malformed bytecode artifact was executed. The
reference evaluator itself and Darwin execution were not requalified here.
