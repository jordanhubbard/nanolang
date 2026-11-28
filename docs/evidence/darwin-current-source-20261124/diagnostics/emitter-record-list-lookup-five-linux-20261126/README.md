# My emitter record-name lookup qualification

I bind `parser.structs` once in `nisa_struct_index`, read its list length once, and compare names directly in the original forward order. I preserve my signature, first-match index, missing-name `-1` and every existing empty-print side effect. I add no persistent cache, parser layout, API, ABI, authority or C product change.

My baseline retains the original production function and the same new useful shadow as my candidate. Removing that shadow reproduces every original source byte. I retain the existing shadow unchanged. My added assertions cover empty/missing/case/prefix/duplicate/last names, same-length alias replacement, alias push and growth to 133 records. Baseline/candidate file SHA-256 values are `691f7f85c0100d2825237946c63d2c5c05b9d1214f24109465b42f403657d2d7` / `34265527a7132cd242acb0f52641be44a31f16099681d7f4d0cbd53b90eee881`.

I bind exact source `cfd62a78d5f8afaa6bbd7cc835d75ca97e3caf00` with one declared file overlay per role. My real source compiler is genuine Stage2 SHA `8c2c8660e05fb97a4bf0017c0de2a9b7f3a7b5fc114128583b62f5cb01dd7fa4`, independently bound to the passing original bootstrap published at `cdcdd3956f2ea632899d964d85383be261a1ea42`.

My standalone comparison completes 270 commands, ten route controls and 240 complete report comparisons: all 24 original sources and expected hashes through C-seed native, C-producer VM/ASan+UBSan native, and NanoLang-producer VM/ASan+UBSan native in both roles. I retain the original 600-second build, 60-second execution and ordinary ten-second shadow limits, including the stock sanitizer/link flags. This comparison driver executes my original fixture and expectations; I do not claim an invocation of the permanent unittest method.

All 3,971 selected source inputs per role, tools, images, immutable hosts, producer, driver and helper guards match at completion. Fresh admission and actual adopted-child cleanup controls pass. Final cleanup leaves no owned child. My archive collector also refuses a real nonterminal attempt before creating its archive.

I inspect the actual retained C translations without rebuilding:

| Producer | Old count/get calls | New count/get calls | Old/new record locals | Old/new record temporaries |
| --- | --- | --- | --- | --- |
| c-producer | 1/1 | 0/0 | 2/1 | 4/2 |
| nanolang-producer | 1/1 | 0/0 | 2/1 | 4/2 |

These are generated call-path observations. My earlier phase and stack evidence motivated the candidate; I establish no whole-compiler speedup, timeout cause or factory repair. A fresh exact bootstrap and original native36/emitter12 gates at unchanged limits remain required. C-owner defects, paired-platform qualification, full pure runtime emission/cutover and all 69 release requirements remain open.

Run `python3 read_archive.py` to verify my 11,045 archived records, raw/compressed bytes, exact Git inputs, qualified compiler origin, old/new source reversal, unchanged reports, materialized C bodies, original deadlines, cleanup, nonterminal refusal and final guards. My manifest SHA-256 is `6ffc195991c8a6dbef1ef9ab53a897887896f79a98fc2542d797d4d7d1d1fdf5`. I preserve prior roots and failures.
