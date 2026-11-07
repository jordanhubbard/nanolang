# My diagnostic-profile input/output preparation

I retain both terminal failures from Linux audit `42398040376` at `06c6d2faf`. Each compiler executes 1,483 commands. GCC completes in 1,074.583 seconds with two missing `managed_runtime_ir.h` refusals. Clang completes in 966.811 seconds with the same two refusals and two coverage-note permission errors. These are observed profile-preparation errors; I do not label them infrastructure or product-runtime failures.

I prepare the header through the existing `managed-runtime-package` Make target in a private `OBJ_DIR`. Its real producer compiles native and wasm32 runtime IR and verifies both through `opt` before publishing a header and source-bound manifest. I prepend the private include path only to its LLVM consumer. I do not copy the shared header or substitute a stub. My real-producer control first observes the missing-header refusal, then successfully compiles the consumer, checks all manifest source hashes and proves the shared build header remains unchanged.

I retain coverage instrumentation and give those compiles private writable assembly outputs. Clang can then write its coverage notes; I remove the complete private directory afterward and report the actual executed argv. My independent coverage control reproduces the `/dev/null` permission refusal before this repair and succeeds afterward.

A host-adapted replay also exposes Apple Clang's ASan builtin `_FORTIFY_SOURCE=0`. I explicitly undefine that builtin before requesting level two, preserving Fortify=2, ASan/UBSan and strict diagnostics. A compiled macro-level control requires the requested value. All six original failing command forms pass on the local host after actual input preparation, output isolation and this explicit reset. I preserve the earlier two Apple redefinition errors separately. Host adaptation adds the actual local Make SDK include paths and installed compiler selections; it is not Linux requalification.

I add LLVM to the optional audit's tool prerequisites, verify the real producer/control paths there, and record selected compiler paths, resolved binary hashes and actual versions. The first audit's tools log recorded default GCC 13 even though the profile detector selected `/usr/bin/gcc-14`; I preserve that attribution limit rather than rewriting the old record.

Exact corrected Linux requalification, the remaining Python-built C profiles, the combined fast-tier budget and full release/native acceptance remain open under #1046 and #976.

My first Linux audit worker records four CPUs, 84,766,449,664 free disk bytes and 15,532,003,328 available memory bytes. Each measurement used two compiler workers. My real private-producer and process-group controls pass on the local host; the shared build header remains unchanged. The local host is CPU-saturated by other qualification work, so I hold additional local compiler fan-out and require the corrected hosted replay.

My default gate passes ten controls (one explicit Darwin/glibc skip), 223 Clang compiles in 53.7 seconds, 223 GCC compiles in 127.2 seconds, the fixture/prerequisite checks and the bootstrap-tool control. These timings include observed local CPU contention and do not replace the hosted budget measurement.
