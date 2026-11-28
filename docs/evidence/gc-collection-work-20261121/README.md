# My completed collection-work observation

I count actual collector work under #1161 using the production packed-only
collector with private counters. I preserve all trial edges, reference-count
operations, the 4096-root trigger and heap layout. The actual complete VM
unit driver passes its 299243 assertions with that observer. The separately
validated CPU/window controller retains its raw-equal hello control, exact
compiler/tool/host guards and automatic stop after 30 seconds of typechecking.

My 27 records refer to one heap and lie inside that window. Between the
first and last record, 28.907403723 seconds apart, I observe:

- 81 complete collection passes and 331776 buffered roots: 4096 per pass.
- 133693815 object visits and 724507731 child-slot inspections.
- 304271 heap allocation calls and 51669922 tracked allocated bytes.
- 50354954 tracked freed bytes and 9780 additional retained objects.

Tracked retained heap bytes rise from 116354888 to 117669856, an increase of
1314968. These counters describe actual repeated graph work and allocation
volume. My instrumentation changes timing, so I do not treat this run as
uninstrumented throughput or full-generation qualification.

I retain the exact private collector, observer VM, original assertion driver,
raw records, phase clocks, sampler controls and complete source/link commands.
My manifest covers 49 logical inputs, 50067539 raw bytes and 13752680 unique
compressed bytes with checked raw/compressed SHA-256 identities.

The separate inlining candidate later reaches program lowering but exceeds
1800 seconds. The leaf-graph candidate exceeds the same limit during
typechecking. Neither produces a Stage2 module. Those full-run terminals are
separate evidence; this receipt contains only the completed work observer.
I use the measurements to test allocation-amortized automatic collection;
explicit collection, correctness and unchanged-budget raw bootstrap remain
required.
