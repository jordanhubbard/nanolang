# My complete 4620 route checkpoint

I retain terminal [run 42452651802](https://github.com/jordanhubbard/nanolang/actions/runs/42452651802) at `4620fff8771fa15d8c3ab7c94e40f1bb635f9cea`. My bootstrap completes both generations, verifies their raw NanoISA modules are byte-identical, and executes the resulting installed compiler with the C seed removed. My complete original 237-case corpus runs on all four routes with zero skips. Ordered testcase class/name identities match the earlier eba checkpoint on every route.

| Route | Passed | Failed | Skipped |
| --- | ---: | ---: | ---: |
| C seed native | 237 | 0 | 0 |
| C seed NanoVM | 234 | 3 | 0 |
| Fresh Stage 1 | 175 | 62 | 0 |
| Fresh Stage 2 | 175 | 62 | 0 |

My C-native result completes only that bounded original-corpus checkpoint. My VM failures are the resource-checker argument transport and the two original generic-context fixtures. The later f92 separator and 78d record-array corrections retain their own focused evidence; this earlier run does not qualify either correction. Both fresh stages already pass those exact generic-context fixtures here. Their complete 62 failure names and diagnostics remain in `route-summary.json.gz`; exact XML, compile/run diagnostics, commands, provider identities and bootstrap output remain in my text bundle. My canonical string focused controls pass all three selected NanoISA producers; the separate full map-context step remains failing.

Artifact 13607829378 passes ZIP CRC checks. My inventory records every original member and hash; `complete-text-evidence.json.gz` restores each retained UTF-8 file verbatim. Native module bytes remain in the private original archive. This direct ZIP does not establish original filesystem modes or symlinks. Per-bootstrap step logs and manifests are not members of this route artifact; I retain its complete bootstrap command output without claiming those absent individual logs.

I leave #1081/#1088 and integrated release acceptance open. Later fresh-stage runs retain the same original corpus and assertions.
