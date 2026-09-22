# My corrected Linux tail and canonical frontend gates

I preserve four separate frozen histories in the adjacent seal and compressed report bundles. Every referenced external content object was rehashed when sealed. The object indexes name retained local CAS paths; the bundles do not contain the full CAS or the inner `/tmp` fixture trees. I retain those fixture paths in their original logs. All recorded phase endpoints have equal source/tool hashes, reaped leaders, absent process groups and no tracked descendants.

| Source | Actual result |
| --- | --- |
| f8ba6a017 ordinary | Fresh backend build passes; all six scalar-tail methods pass in 9.313 seconds, including generated C and LLVM O0/O2 under 512 KiB stacks. Eleven source methods pass before the bare Nano emitter refuses the transitive-import fixture. |
| f8ba6a017 sanitizer | Fresh backend build and all six scalar-tail methods pass in 15.003 seconds. Generated C uses ASan/UBSan/LSan; generated LLVM native code uses explicit ASan. VM, lli and Wasm remain ordinary. |
| bdf19160c canonical | C owning gate passes 2,422 checks. All three canonical nanoc_v06 builds pass using the separately retained C-seed/Stage1/Stage2 compilers. The first unchanged byte program then fails canonical checking at `let d: byte = 7`. |
| 873b79ad2 alias | The first C-seed canonical build stops during imported shadows. Static inspection identifies the old lowering fixture function named `byte`, which is now a reserved type alias. I rename that fixture in d002ac238 without changing payload or assembly assertions; its fresh gates are separate. |

I do not claim a fresh native bootstrap, full source matrix, Darwin tail acceptance, shared capture acceptance, or release readiness from these bounded results. My bare emitter and canonical frontend histories remain distinct. The reports retain original limits and first failures.
