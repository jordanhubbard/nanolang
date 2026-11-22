# My bounded native capture comparisons

I retain successful diagnostic [45018368924](https://github.com/jordanhubbard/nanolang/actions/runs/45018368924) at `919b001b5b7611395b432175d66fd4270cd41b18`. I run the unchanged original `test_clang_external_assembler_cache_restored_inputs` method in nine native configurations on one worker. Each method exercises all four original kind/shared cases, their answer/restoration/reuse/count assertions and original per-command deadlines. These are nine focused observations, not my complete128-method source gate or production admission.

| Probe build | Helper build | Backend | Wall seconds | Child CPU seconds |
|---|---|---|---:|---:|
| Historical mixed | GCC | supported2.42 | 3.009 | 2.952 |
| GCC | GCC | supported2.42 | 3.000 | 2.970 |
| GCC | GCC | fixedr1 | 2.996 | 2.967 |
| GCC | Clang | supported2.42 | 3.112 | 3.082 |
| GCC | Clang | fixedr1 | 3.731 | 3.087 |
| Clang | GCC | supported2.42 | 3.021 | 2.978 |
| Clang | GCC | fixedr1 | 2.993 | 2.964 |
| Clang | Clang | supported2.42 | 2.991 | 2.960 |
| Clang | Clang | fixedr1 | 3.111 | 3.081 |

I measure each configuration once, in the retained order. I do not claim statistical equivalence or assign a cause for the earlier full-run timeout. This case demonstrates no large tool-axis CPU penalty: child CPU ranges from2.952 to3.087 seconds. The3.731-second observation includes more wall time without a comparable CPU increase. Each configuration executes45 observed child commands, including eight actual builder calls; all405 return zero.

My worker reports AMD EPYC7763, four available vCPUs, Ubuntu image `20261116.202.1` and an Azure kernel. Before/after shell controls take1.2126/1.2130 wall seconds at99.96/99.97 percent child CPU/wall share. These observations describe this worker and interval only. Earlier complete success uses an AWS kernel/image `20261116.88.1`; the failed combined consumer uses Azure/image `20261116.202.1`. Those labels do not establish a causal platform difference or earlier CPU availability.

My [manifest](manifest.json) retains27,401 logical files,256,735,092 raw bytes and448 new packed payloads totalling1,003,933 bytes. It includes every copied source input, actual native probe/helper/backend images, original temporary fixture products, commands/streams, source maps and stable loader closures. Existing immutable payloads remain shared by content hash. My [independent reader](read_archive.py) verifies every raw/packed hash and length, all26,190 source/native input identities, all405 observed commands and every reported native tool image without executing ELF on the reader host. A separate actual Darwin native timing control validates the later streaming observer only.

I next compare the original split assembler-search/recovery method, whose Python compiler-wrapper and phase paths are absent from this fast cache case. I preserve its helper assertions, argv and per-command deadlines and retain command start/completion/exception measurements immediately. Production C, tool admission/defaults and ordinary CI remain unchanged. My complete Clang/r1 pair, final platform/current-head acceptance and whole release remain open.
