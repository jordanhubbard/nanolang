# My original assembler-search cost observation

I retain successful bounded diagnostic [45018884695](https://github.com/jordanhubbard/nanolang/actions/runs/45018884695) at `818d0126b308a0cdb5c3ac44053ea396e2fd6f0a`. I execute the unchanged original split assembler-search/recovery method once with the historical mixed/supported2.42 configuration and once with all-Clang/fixedr1. Both execute all12 placement/external/shared cases, including failed replacement and recovery assertions, with every original target argv and per-command deadline intact.

| Configuration | Wall seconds | Child CPU seconds | Observed commands |
|---|---:|---:|---:|
| Historical mixed/supported2.42 | 44.400 | 42.059 | 292 |
| All-Clang/fixedr1 | 41.870 | 41.447 | 292 |

Each method has72 builder calls, including12 expected failed publication attempts. My observer streams292 command starts and292 completions per configuration. I retain all original temporary products, source/native images, before/after source maps and stable native loader closures. All original assertions pass. These single ordered observations demonstrate no large pair-specific cost in this method; they do not establish statistical equivalence or complete source acceptance.

My all-Clang/fixedr1 builder calls consume21.665 CPU seconds across48 no-package calls and17.870 seconds across24 six-package calls. Captured native tool-run phases total3.707 seconds and tool-query phases2.485 seconds. These partial phase totals do not explain all builder CPU. I measure package overhead as one component and assign no sole cause.

This worker reports Xeon8573C, four available vCPUs, Ubuntu image `20261116.202.1` and an Azure kernel. Before/after shell controls use1.0338/1.0175 wall seconds at99.95/99.97 percent child CPU/wall share. These controls describe this worker and interval only.

My [manifest](manifest.json) retains7,157 logical files,79,478,099 raw bytes and355 new packed payloads totalling215,539 bytes. My [independent reader](read_archive.py) verifies every raw/packed hash and length, all5,820 source/native input identities,584 observed commands,1,168 command events, original expected recovery refusals and actual native image hashes without executing ELF on the reader host. My separate Darwin native timing control validates the untraced observer path only.

I next trace one package-bearing build's process/open/read operations in a fresh original method observation. I retain the original target argv and per-command limits, avoid expanding child environments, and request raw read arguments so the trace records no read buffers. This is localization before any runtime or fixture change. I repeat no complete128-method source corpus or upstream producer/GAS build, change no tool/default admission, and keep the full Clang/r1 and release gates open.
