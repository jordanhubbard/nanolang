# My Stage2 CPU observation and packed-leaf repair

I investigate #1161 against the actual Linux compiler and tool snapshot at
`74644c29f`. My authoritative 1800.135-second Stage2 failure remains in the
[original receipt](../bootstrap-stage2-timeout-20261121). I preserve its source,
VM and six-library host closure. None of the diagnostic runs below replaces
that qualification or changes its 1800-second generation and 30-second
complete compiler-shadow budgets.

My completed direct main-thread CPU observation uses the original VM executable
and actual raw Stage1 compiler. A private preload collector samples its thread
CPU clock every ten milliseconds, records the interrupted program counter and
monotonic timestamp, and writes fixed records without allocation or unwinding
in the signal handler. Actual executable maps and ELF load segments resolve
addresses through the original images. Finite and intentionally terminated
controls record 49 and 80 samples; the real hello control records 37 samples
and produces byte-identical compiler output.

I automatically stop only this diagnostic after its independently recorded
30-second typecheck window. All 2997 samples resolve. Of these, 1714 are in
`for_each_child`, 563 in `scan_black_child` and 550 in `mark_gray_child`:
2827/2997 (94.3%) lie in collector graph traversal/restoration. This measures
the first 30 seconds of typechecking, not the entire historical timeout.

I retain three different observations that do not establish that CPU result:

- My private native `-pg` Stage1 self-compile exceeds 1800 seconds during
  shadow lowering. Its hello control produces a readable call-count graph
  without accumulated timing ticks. Neither supplies a full target profile.
- My sparse dispatch sampler records 270209 complete checkpoint rows. Its
  growing-log observer fails to return after the completion event; I stop
  only that private process group 171.4 seconds later. Analysis excludes the
  overrun and retains 13327 checkpoints in the first 30 seconds, occupying
  only 277/3000 ten-millisecond bins. Dispatch counts are not CPU attribution.
- Both gprofng controls warn that their collection timer period changed to
  zero and their profile data may be unreliable. I preserve the experiments
  without using their samples to diagnose my compiler.

My source review finds a concrete collector defect: packed INT/FLOAT/BOOL/U8
arrays enter the cycle-suspect buffer even though their representation has no
heap edges. At `208441b46` I exclude only these proven leaves. I preserve the
trial-deletion algorithm, boxed cycle handling, heap layout and 4096-root
trigger. My two regressions check immediate final-release reclamation and
allocation-free alias churn beside an independently retained boxed cycle and
a large live graph.

Against the original providers, my baseline complete VM unit driver records
299238 passes and exactly those two new failures. The repaired collector
records 299243 passes and no failures. My targeted ASan/UBSan run, with leak
checking, also passes all 299243 assertions and the unchanged original
allocation-failure driver. I instrument the actual VM, heap, collector,
values and test driver; unrelated provider objects remain their original
prebuilt objects. This is not an all-provider sanitizer claim.

My manifest retains 175 logical files, 303,240,665 raw bytes and 52,357,451
unique compressed bytes with checked raw/compressed SHA-256 identities. I
retain actual diagnostic binaries, controls, raw records, failed observer
state, manifests and used controller sources. Full unchanged-budget compiler
generation, exact raw equality, fresh authoritative bootstrap and Darwin
acceptance remain separate gates. These unit and CPU observations alone do
not prove that the Stage2 deadline is repaired.
