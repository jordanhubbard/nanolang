# My fresh checked-emission phase checkpoint

Run [42492701660](https://github.com/jordanhubbard/nanolang/actions/runs/42492701660) at `9c6312a3d6ab42e65b43d34385a34052a38c8e42` finishes the original CLI step successfully and continues into the complete corpus. Every original C-seed, VM, Stage 1 and Stage 2 CLI route produces the unchanged 2/0/1/2 outcomes, without timeout, under the existing 1800-second compilation bound. I do not report the still-running corpus as passed.

My two fresh stage compilations take 584.716050 and 579.634008 seconds. All 22 markers per stage are present exactly once and ordered by the same monotonic clock; all 18 observed tool calls per stage complete successfully with unchanged tool/input hashes. I retain every original metadata record in deterministic gzip and derive `observations.json` from those records.

| Marker interval | Stage 1 seconds | Stage 2 seconds |
| --- | ---: | ---: |
| Imports | 64.120101 | 63.586762 |
| Lexer | 54.813377 | 54.480964 |
| Parser | 23.521918 | 23.234793 |
| Typecheck | 55.473855 | 54.313340 |
| Program lowering | 93.080781 | 92.001394 |
| Shadow lowering | 161.237625 | 160.286046 |
| Shadow assembly | 20.469165 | 20.453879 |
| Shadow execution | 1.064064 | 1.053420 |
| Product assembly | 4.717374 | 4.722027 |
| Native publication | 103.218729 | 102.517771 |

These are wall intervals, including work nested between their boundaries and marker observer overhead. I do not label the shadow-lowering interval as assertion execution. Its 160-second cost is separate from the one-second shadow-execution interval and the C interpreter's ten-second creation gate. The typecheck/program/shadow split narrows the next investigation; it does not yet identify an inner function or allocation cause.

Across all observed calls, translator elapsed time sums to 42.829424 / 42.754969 seconds and C-tool time to 60.416347 / 59.780021 seconds. These sums overlap their enclosing marker intervals and must not be added to them. Inter-marker gaps, entry/return work and observer overhead remain explicit in the retained records. I make no before/after performance claim across differently loaded runs or later source changes.

Artifact 13626145178 is a 131028-byte metadata ZIP. `archive.json` records its digest and successful CRC; all 121 entries are retained as exact text and hashed inventory. It does not contain the copied executable/runtime bytes needed for independent product custody. I require the later full artifact for that check, and retain full original corpus, current-source/platform, native SDK and release acceptance separately.
