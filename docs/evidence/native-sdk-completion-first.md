# I retain both-host completion controls and distinct bootstrap outcomes

I qualify source `bae599477588b57b89159e10f03d5f5518ca6093` with fresh independent inputs and ordinary compiler providers. Both hosts pass the actual completion decision/pipe probe ordinarily and under ASan/UBSan, then a freshly built corrected C seed passes actual success, assertion refusal and a safely supervised running-loop deadline with preserved output sentinels. Sanitizer scope is the decision/pipe probe, not the complete compiler provider closure.

| Host | Phase | Exit | Seconds |
| --- | --- | --- | --- |
| linux | decision-ordinary | 0 | 2.480 |
| linux | decision-sanitized | 0 | 2.516 |
| linux | corrected-cseed | 0 | 23.618 |
| linux | actual-completion | 0 | 3.208 |
| linux | build | 0 | 59.989 |
| linux | bootstrap | 2 | 60.083 |
| puck | decision-ordinary | 0 | 3.008 |
| puck | decision-sanitized | 0 | 3.266 |
| puck | corrected-cseed | 0 | 10.224 |
| puck | actual-completion | 0 | 3.506 |
| puck | build | 0 | 73.980 |
| puck | bootstrap | 0 | 341.334 |

Darwin's original complete bootstrap passes, preserving C-seed/Stage-1/Stage-2 products. Linux's first full bootstrap still refuses at the unchanged ten-second shadow deadline, after its component build passes. No outer phase times out, and all recorded groups/descendants are absent. Source/tool endpoints remain equal. I do not infer the Linux active shadow or delay cause from Darwin's pass; no dependent Linux corpus executes.

My [manifest](native-sdk-completion-first/manifest.json) preserves all 267 reports, probe artifacts and driver/copy records, including both source/tool/product endpoint inventories. Reports over 40,000 bytes use lossless gzip; binary images are retained in the named immutable content store. Original source/provider trees remain retained at inventory paths. The remote Darwin report copy is independently checked for bytes, length and modes. Full paired/installed SDK and Linux bootstrap acceptance remain open.
