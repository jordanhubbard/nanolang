# My isolated test-report formatter comparison

After complete published foundationb8cf8018ec and fresh idle admission (two CPU shares above99.98%, unchanged source/tool/host/provider/artifacts, adequate local storage/memory), I compare one private test formatter change at base7b75651234. I move the same256 exclusion-bit concatenations into a bounded string before appending to a growing report. Every old assertion/product source remains unchanged.

All20 commands and72 complete reports pass: original full-stack load-4-True-True-256 plus eight independent C single-bit reports, across old/new C-producer VM and sanitized native, in two counterbalanced rounds. Original600 translation/build,60 full-stack,8-case600 baseline and default10 shadow limits remain. No fresh bootstrap or compiler pin changes occur.

| Product | Old full-stack seconds | New full-stack seconds |
| --- | --- | --- |
| C-producer VM |8.124,8.041 |0.417,0.425 |
| C-producer sanitized native |19.287,19.577 |2.876,2.916 |

These timings describe this test-report case on this Linux host. They establish no compiler-wide gain, whole-corpus time, driver integration or release acceptance. Current/genuine-source/full-corpus/paired qualification remains required before accepting updated drivers. The frozen foundation and earlier downstream roots remain unchanged.

I verify this evidence with:

```sh
python3 docs/evidence/darwin-current-source-20261124/diagnostics/affine-report-benchmark-linux-20261125/read_benchmark.py
```

My manifest is `3d98ddb9bf7c4c28e8258a9bb8f10664edb0ee308ee275b7e75537e5204c0ce8`. The reader verifies1598 records, publication/admission, exact original argv and overlay, every report/guard, and the two observed faster pairs in both products.
