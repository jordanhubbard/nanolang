# I link my production tracing dependency

I qualify source `829be95a26d4391f0f067402b893c8f6b99213d4` under #1208,
after the live-value/tracer correction in [my preceding receipt](../tracing-assignment-linux-20261123/README.md).
My first actual isolated bin/nano build with NANO_INTERPRETER fails to link
trace_* and g_tracing_config because my common source list omits tracing.c.
I retain that failed build/root. I add the actual source and filter its common
object out of the observer target, which supplies an explicitly enabled tracer.
My reader reconstructs these two Make edits exactly.

Three fresh interpreter products pass: tracing-enabled ordinary,
tracing-enabled ASan/UBSan, and default ordinary builds. Each actual product
runs the same assertion-bearing string/callable/record program and prints
TRACE_PRODUCTION_OK; help and version also pass. All three profiles pass the
focused enabled/filtered/disabled tracing fixture, checking21 live old/new
values,7/1/0 assignment events and all14 runtime-kind labels. The instrumented
focused fixture keeps leak detection enabled.

Both tracing-enabled profiles pass all134 original interpreter and88 checker
methods, the strict zero-name callable audit,991/991 binding regression,
complete expression/globals fixture, borrowed-value and environment controls.
I disable leak detection explicitly for the broad CLI/interpreter/checker
runs. Focused sanitizer fixtures keep it enabled; the complete original
record/union fixture still fails with3720 leaked bytes in159 allocations.
I retain that entire failure and leave #1116/#1140 open.

My standalone CLI still does not configure its documented tracing flags or
run shadows automatically. I preserve that complete interface under #1209
and correct FEATURES.md's completion claim without removing any requirement.
Two actual native controls reproduce the gap: --trace-all is treated as a
filename and exits1, while a program whose shadow asserts false still prints
UNEXPECTED_SHADOW_ACCEPT from main and exits0. These are retained contract
failures. A linked tracer, API regression or ordinary product exit0 does not
accept the standalone trace/shadow interface.

I verify my lossless source/products/archive with:

```sh
python3 docs/evidence/tracing-production-linux-20261123/read_archive.py
```

My reader passes1505 records and1389 exact Git inputs, including all35 current
qualification observations and both separate CLI contract failures. It verifies
compressed/raw hashes, exact Make reconstruction, unchanged interpreter/eval/
tracer/test source, all actual Linux ARM64 products, tool-image guards and the
complete original failures. My recorded host profile identifies GCC13.3.0 and
the compiler/assembler/linker/sanitizer images; it is not generic toolchain
admission. Identical objects are shared with the preceding hash-bound archive.
The source, observers, commands, streams and private product paths remain
preserved. The preceding receipt retains the original syntax/enum refusals,
my invalid first Make insertion and deliberately late-serialization ASan UAF.

I change no Puck-owned nvm2c/merge/regex/capture/checkpoint/CI source or retained
compiler/VM/AOT image. All frozen roots/audits/stashes and NAS/TCC/migration
pauses remain intact. I perform no NAS action, claim no historical timeout
cause or performance gain, and leave paired current-source/full release
acceptance open.
