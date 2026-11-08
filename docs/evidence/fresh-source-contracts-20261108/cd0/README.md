# My fresh cd0 source checkpoint

I retain early artifacts 13620730221 (focused) and 13620457786 (source metadata) from [run 42483191785](https://github.com/jordanhubbard/nanolang/actions/runs/42483191785), source `cd0a4aa0a2df7fd28159523a0efc87458dde7406`. Fresh bootstrap succeeds and all nine canonical methods pass through each of three producers; all six map controls pass.

My 92-method source suite reports two failed provider subcases in one method. Both fresh stages correctly refuse U8-plus-string with `string concat needs two strings`, but that message does not match the existing `(type|operand|argument|arity)` diagnostic assertion. The negative cases refuse before replacing output. I preserve the failure and leave the assertion unchanged; I state the exact string operand requirement in my diagnostic and add a useful emission shadow.

The unchanged whole byte fixture and scalar arithmetic controls now pass through both fresh stages, as do all anonymous-record controls, including the unchanged shadow-only literal fixture and the separate production-main AGG_PACK/AGG_GET control. Character, directed scalar argument and phase-marker helper methods pass. These are early source results, not later-source, original corpus-budget or release qualification. The CLI/corpus portions of this same run remain live when I retain this checkpoint.

Both ZIP CRCs and all file hashes verify. The deterministic compressed text records retain all five source metadata and ten focused files. This early upload supplies hashes/logs, not compiler or generated native binary custody. Private originals remain under `~/.worktrees/nanolang/audits/2026-11-07/cd0-early/`.
