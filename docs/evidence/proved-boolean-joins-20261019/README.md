# I normalize my proved boolean join carriers

I reproduce hosted coverage failure `39224053627` with my unchanged fresh `16eb3b1f9` Stage 2 producer. Emitter module generation succeeds in 513.093 seconds, including its shadows and entry assertions. The original translator refuses function 589 (`purity_identifier`) at join 213. Diagnostic-only emission identifies slot 0: saved VALUE carrier 8, incoming BOOL carrier 9, solved exact BOOL shape 7. My smaller empty `array<Flag>` source reproduces the same defect at function 1, join 55.

I normalize a boxed edge only when the solved destination is exact BOOL, using the existing runtime tag-checking extractor and a fresh edge-local temporary. I preserve conditional fallthrough, temporary high-water accounting, parallel join assignments and OPTIONAL boxing. I do not accept integer truth values as booleans. Independent read-only review found no blocker.

The unchanged full emitter module translates successfully after this correction in 15.907 seconds. Fourteen scalar/optional-record methods pass in 66.053 seconds. Expanded source qualification passes in 50.461 seconds through NanoVirt and both fresh installed generations: both operand orders, OR/AND truth tables, RHS evaluation counters, actual checked extraction in empty-array modules and O0/O2 ASan/UBSan execution. Additional generated-C probes execute the actual boxed-record join with BOOL values and reject INT-tagged 0/1 under NDEBUG. I retain the original failed guard-harness cleanup-name attempt and its correction; no product assertion was weakened.

After two worker recovery samples, I build the full native emitter in 16.739 seconds using the same compiler/link flags as my retained bootstrap native step, then execute its entry assertions successfully in 0.249 seconds. The hosted coverage rerun remains pending; this local ordinary-profile execution does not claim Linux coverage instrumentation. Original module/C artifacts remain at their indexed paths with hashes. This bounded fix does not complete #976 or 5.1.

I also pass my serial strict gate: 222 Clang and 222 GCC 16 compilations, with zero failures.
