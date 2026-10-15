# My paired WebSocket source checking

I build on `044bf7504`. I extend immutable source catalog queries, both parsers,
C/Nano source plans, namespace selection, nominal types and body checking to the
exact WebSocket contract. I retain string fields and three input parameters for
send, and construct Message records with complete named bool/string fields.
I do not enable WebSocket executable lowering or product host grants here.

My source-plan corpus compares ten WebSocket cases through independent C and
Nano implementations, including aliases, permuted bindings, all three catalogs,
wrong views, missing/wrong methods, version mismatch and legacy File refusal.
Actual C-seed, Stage 1 and Stage 2 products agree byte-for-byte on rows and run
all selected shadows. LLVM instrumented providers and the original C ownership,
allocation and budget tests pass. Existing mixed File/TCP plan cases also pass.
The installed compiler stages compile the changed test/module source; this is
not a fresh whole-compiler fixed point.

My source-input consumer gate compares all three rendered catalog views and
runs C-seed native plus NanoVirt/Stage1/Stage2 VM and translated-native products.
It passes in 24.185 seconds. The WebSocket plan gate passes in 22.067 seconds;
its File/TCP/C-allocation adjacency passes in 39.773 seconds.

My independent body probes use the C checker and current Nano checker source
compiled into VM and sanitizer-instrumented native products. Both WebSocket body
methods pass in 40.883 seconds: complete lifecycle, reordered Message fields,
strings, all Result payloads, malformed fields/signatures, pure effect rejection
and fabricated-owner refusal. Both ownership methods pass in 42.031 seconds:
loops, branches, borrowed helper calls, consuming close, leaked results and
connections, double-close and branch disagreement. Existing driver probes retain
prior output when executable lowering refuses the WebSocket catalog.

I retain initial failures: a missing corpus-local alias helper; the C parser's
still-File/TCP-only declaration selector before I extend both parsers; and a TCP
adjacency regex that omits the existing connection-policy diagnostic. A second
fixture still expects the old unimplemented-TCP message when it supplies only a
File grant; I require the current explicit TCP-policy refusal instead. I preserve
type/ownership statuses and output checks while correcting that test oracle.

All six TCP body/ownership methods now pass across the retained runs: four
methods in the first corrected run and the two remaining methods in the final
42.626-second focused run. I retain the failed old-message assertion rather than
claiming the earlier combined run succeeded.

Public WebSocket real traffic still requires a host permitting localhost bind.
Paired executable lowering, host-grant product selection, public DNS integration
and final release/platform acceptance remain open.
