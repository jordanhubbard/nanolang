# My measured fresh driver phase checkpoint

I retain early CLI artifact 13621337813 from [run 42482810275](https://github.com/jordanhubbard/nanolang/actions/runs/42482810275), source `f4966d8242286c735fb11dd4bc82c03aafa33946`. All four actual providers compile and execute the unchanged CLI contracts with exact statuses 2/0/1/2. My selected 1800-second compile allowance remains unchanged; this does not satisfy the separate original 120-second corpus gate.

| Measured interval, seconds | Stage 1 | Stage 2 |
| --- | ---: | ---: |
| Outer compile wall | 538.176 | 541.930 |
| Imports | 62.046 | 62.007 |
| Lexing | 53.879 | 54.106 |
| Parsing | 22.681 | 22.795 |
| Parse completion to checked emission | 1.702 | 1.747 |
| Checked emission | 269.632 | 269.008 |
| Shadow assembly | 19.963 | 19.987 |
| Shadow execution | 1.044 | 1.043 |
| Product assembly | 4.634 | 4.654 |
| Native publication | 101.699 | 105.697 |
| After driver-return marker | 0.598 | 0.589 |

Both stages retain all sixteen ordered driver markers inside their actual outer monotonic compile bounds. Each also retains eighteen completed tool observations: seventeen CC invocations and one nvm2c invocation. Stage 1 measures 42.339 seconds translating and 59.371 seconds across CC calls; Stage 2 measures 46.687 and 59.022 seconds respectively. Small CC metadata probes occur inside checked emission. The actual final CC command retains C11, O1, my runtime object and original linker flags. Every tool exits zero, retains unchanged input hashes and matches its before/after executable hash.

In this instrumented run, checked emission is the largest measured phase. The after-return interval is small, so I do not attribute hundreds of seconds to process cleanup. Checked emission still combines type checking, program lowering and selected shadow lowering; I must measure those parts before choosing an optimization. These are wall intervals including observer work, not CPU samples. I do not infer the cause of my older c003 uninstrumented timings from a different source pin, and the extra marker calls can change allocation/collection timing. No flags, shadow selection, assertions or deadlines change here.

The C seed and NanoVirt compile in 37.946 and 4.483 seconds. I verify every original case, unchanged provider/fixture hashes, marker bounds and tool histories in `verified-phase-analysis.json`. ZIP CRC and all 109 file hashes verify; I retain every file as exact UTF-8 text in deterministic compressed form. The ZIP is 125478 bytes, SHA-256 `a7b02b11eb41f731c26c1d4249f16c305d673afdcdc07545cda21926a4f2860f`, privately retained under `~/.worktrees/nanolang/audits/2026-11-07/f496-cli/metadata.zip`. This early metadata supplies observed product hashes, not program/compiler/runtime bytes. Full product custody, original corpus and later-source/platform/native lifecycle admission remain separate.
