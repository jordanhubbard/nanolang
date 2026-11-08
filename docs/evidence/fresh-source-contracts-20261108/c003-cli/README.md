# My measured fresh native CLI checkpoint

I retain the early CLI metadata from [run 42475960040](https://github.com/jordanhubbard/nanolang/actions/runs/42475960040) at `c003d0d9776cbef3368ee685fe2260b24a7cd8e5`, artifact 13616994193. All four fresh providers execute the unchanged helper contracts with exact statuses 2/0/1/2 and the ownership diagnostic. My selected compilation budget remains 1,800 seconds; this does not satisfy the separate original 120-second corpus budget.

| Provider | Compile wall seconds | Translator seconds | All observed CC seconds | Outside observed tool calls |
| --- | ---: | ---: | ---: | ---: |
| NanoVirt | 3.403 | — | — | — |
| C seed | 31.983 | — | — | — |
| Stage 1 | 395.594 | 37.108 | 49.916 | 308.570 |
| Stage 2 | 393.340 | 37.240 | 50.003 | 306.097 |

Each stage retains 18 completed tool observations. The selected native C invocation preserves `-std=c11 -O1`, my runtime object and original linker arguments. Every observed tool exits zero, retains unchanged input hashes and matches its before/after tool hash. Both stages report identical generated C (9,691,751 bytes, SHA-256 `c19cf49f8cd3261c2c3481374167dad3bb38d4d3cb3e7d33fd99eb07edcd4ba0`) and native output (14,406,752 bytes, SHA-256 `a6152aa9a7588bc55002254e81ad82bec6bdc5cee92c21a133ae845fc2b54aaa`). These early records contain hashes, not those product bytes. Full product custody remains dependent on the final artifact. This first observer revision also predates direct runtime-object copying.

I retain all 77 uploaded files as exact UTF-8 text, a complete SHA-256 inventory and the verified ZIP CRC and digest. The table's tool columns sum subprocess elapsed time, including small module-build commands in the CC column. The outside column subtracts those sums from outer compile wall time; observer startup/copy overhead remains included there. I do not attribute it to frontend work, shadows, cleanup or infrastructure without further measurement. The observed C compiler alone does not account for most of this build time.

My original full corpus continues in the same job. This checkpoint does not qualify later anonymous-record, byte-alias, character or directed-argument source changes, a complete release candidate, or full native lifecycle admission.
