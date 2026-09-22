# My Make-to-Python native link closure

I qualify the one-line recipe correction at `585a6bc1219f7768f3fff59477ed9a96374cf9e6`, based on canonical main `87e0cef0e`. I pass the expanded Make `CC` and `LDFLAGS` explicitly to the existing four-module Python suite. My native helper still prefers `NANO_NATIVE_TEST_CC`. No fixture assertion, compiler warning or global export policy changes.

My [original hosted failures](../pr945-hosted/README.md) remain separate. The corrected source runs the actual `make -j2 test-nanoisa-src-nano` target; the launch environment removes inherited `CC`, `CFLAGS`, `LDFLAGS` and `NANO_NATIVE_TEST_CC`, then supplies the selected compiler and flags only through Make arguments.

| Fresh configuration | Actual result |
|---|---|
| Ordinary compiler and providers | 90 methods pass; 212.944 seconds including preparation |
| Coverage compiler and providers | 90 methods pass; 239.411 seconds including preparation |
| First fully instrumented ASan/UBSan preparation | Stops after 109.961 seconds: `nanoisa_emit` shadow execution reaches the original 60-second deadline, before the suite |
| Ordinary providers plus freshly ASan/UBSan-instrumented AOT runtime | Preparation passes in 56.725 + 0.675 seconds; actual Make target passes all 90 methods in 192.644 seconds |

I preserve the first timeout without increasing its deadline. The last row supplies the instrumented runtime's actual object dependencies through Make's `AOT_RUNTIME_OBJECTS`; the compiler, VM and other providers remain ordinary. I retain hosted `detect_leaks=0` and strict UBSan halt behavior, so this is not LSan or fully instrumented-provider acceptance.

I inspect the actual AOT object: ordinary has no gcov/ASan/UBSan references; coverage imports gcov; the mixed configuration imports ASan and UBSan. In the mixed run, the exact runtime hash remains unchanged across acceptance. Retained child commands show each of the three previously failing links receives the expanded sanitizer link flags, produces a retained native executable, and executes it successfully. I preserve all original 90 methods and their assertions.

My six phase terminals contain five passes and the first preparation failure, with leaders reaped, process groups absent and no outer timeout. Twelve source/tool pairs match. My report bundle contains 4,030 exact members; its manifest hashes each raw report. My local immutable CAS retains 2,901 objects / 1,022,109,475 bytes. The report maps contain 9,457 product references, including 7,908 in post-phase product maps. The CAS lives at `/home/jkh/nanolang-qualification/pr945-make-flags-seal/objects`; source snapshots and before/after maps are retained beside the original qualification roots.

The ordinary and coverage fixtures cleaned their temporary native products according to their original behavior. I retain their providers and raw successful results; I do not reconstruct those deleted files. The mixed observer records subprocess arguments/results and retains fixture temporary directories. It changes no assertion or command deadline, and does not recover compiler-owned transient files. My raw observer/driver sources are bundle members.

This repairs task `task_96a578e4e4d6f645c396c0a7c453a677`. Actual merge and corrected hosted acceptance remain separate. PR945 keeps its original head and failed checks until its own integration decision; full 5.1 remains open.
