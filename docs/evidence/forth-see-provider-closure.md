# I restore the Forth SEE provider closure

I qualify the six-line manifest correction at `094b846ba`. I add existing mixed-FLOAT and File metadata providers required by sources already in the Forth SEE manifest. I change no runtime authority, interpreter source or disassembly behavior.

On Linux I build a fresh C seed, then compile the complete unchanged Forth interpreter with its normal shadows in 8.384 seconds. On Darwin I build a fresh C seed with the selected Xcode compiler/SDK in 6.100 seconds, then compile the same source and shadows in 12.597 seconds. Both builds and compilations exit zero.

On both hosts my focused Forth file calls built-in `see dup`, receives the expected missing-bytecode diagnostic through `nl_forth_see`, displays the user-defined `square` body and evaluates `3 square` to 9. This establishes source/module linking and the SEE call boundary; it does not establish successful disassembly of an existing interpreter bytecode image.

My first Linux demo exits zero but reports a nonempty stack. I preserve that terminal and keep `task_c56c63dd12934b08962107d5ee5e627b` open. I do not call the full demo accepted or repeat it in the Darwin gate.

My [reports and hashes](forth-see-provider-closure/report-sha256.json) retain the commands, outputs, source/compiler inputs and summary. All38 selected Linux provider/compiler hashes remain unchanged. All6652 Darwin archived source files match the exact commit and remain unchanged across execution. Darwin compiler/SDK selection was captured before execution; Linux compiler tool identity was inspected afterward, so I make no Linux before/after tool-identity claim.

Local artifacts remain in `/tmp/nanolang-forth-provider-qualification`; Darwin artifacts remain in `/private/tmp/nanolang-forth-provider-evidence-094b`, with sources in `/private/tmp/nanolang-forth-provider-094b`. The original source maps are retained there. I close only the provider child after canonical merge; full Forth, File and release acceptance remain open.
