# My isolated JSON span-writer comparison

I compare my mechanically copied current formatter against a private candidate that writes contiguous ordinary bytes with fwrite. I preserve the trace-event body and its per-boundary flush. Production src/eval.c remains unchanged. My fixed plan runs twenty alternating pairs, forty total replays, with the same complete 1,286-start/complete-event fixture.

All forty uncounted replays produce the same 550,034 bytes. Original median time is 0.811209 seconds (quartiles0.774402/0.864839); candidate median is0.037946 (quartiles0.034583/0.042702). Every paired candidate replay is faster on this worker. These are current regular-file measurements, not a comparison with an earlier worker session and not a whole-compiler gain. A separate counted image reports292,890 ordinary-byte calls versus5,144 span calls; I do not use counted-image timings for the comparison.

I retain six actual native Mach-O executable images locally under .test_output/shadow-json-span-comparison-first/. Both ASan/UBSan images and all four plain/counted images pass the original255 nonzero-byte, empty/null/embedded-NUL, UTF-8, long-string, full-event, boundary-flush and failed-stream-visibility controls. Source, selected launcher and image hashes remain stable. Compiler/SDK binary closure and equivalence under every partial-write fault are unqualified.

I retain every operation, sample, pair, source/helper/tool/image identity and control output. My original helper exactly matches current src/eval.c and the earlier retained formatter. This private candidate does not change selected shadows, budgets or source compilation. Adoption still requires reviewed stream/error semantics, actual whole-factory comparison and original complete source/platform gates.
