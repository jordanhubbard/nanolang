# My measured Apple checker-shadow checkpoint

Run [42494678940](https://github.com/jordanhubbard/nanolang/actions/runs/42494678940) at `c6f06909c8d7e11edb6baac0f91be41311907391` enables the existing `NANO_SHADOW_TRACE` only during checker creation. My selected compiler/SDK, deadline, flags, all selected dependency shadows and assertions remain unchanged. The preceding untraced 4f timeout remains retained separately; a subsequent pass does not explain its cause.

All 687 dependency/root shadows complete without failures. Their recorded body durations sum to 9.714615 seconds; this sum excludes selection/setup/reporting and is not the parent-observed complete interval. `tc_validate_layouts` takes 1.286388 seconds, checker `main` takes 1.271700, and the last printed formatter from the earlier timeout takes 0.000548. I retain all 1374 ordered start/completion events in `shadow-summary.json`. These durations identify measured costs in this passing attempt; they do not retroactively attribute the failed attempt.

Checker creation and execution succeed, followed by all 128 integration methods in 116.009 seconds. The independent original translator gate is still live when I retain this early checkpoint. Artifact 13623580040 passes ZIP CRC verification, and all 2188 files are exact UTF-8 text with retained hashes. This early upload records checker hashes but excludes its actual binary bytes; full Mach-O product custody, complete gates and final platform qualification remain pending.

Under #1065 I keep the ten-second margin concern open. I do not raise the deadline, remove a shadow or call the original failure infrastructure. Under #1109 I keep full Apple/fresh-stage acceptance separate until the remaining gates and actual products verify.
