# I preserve caller-owned borrowed values

I qualify source `0dd615a03871c70fd16e6a36b8e98013f47cc969` under #1207,
#1116 and #1140. I keep borrowed environment slots non-owning during string
alias accounting, rebinding and static-array root traversal. My owning roots
still reclaim an array exactly once beside shared and mutable aliases. I change
no public layout, source borrow permission, original assertion or deadline.
I do not enable record-literal or callable-field reclamation.

My two original probes fail against retained fully ASan/UBSan-instrumented
baseline providers: rebinding a borrowed callable frees its caller's name and
signature; environment teardown frees a caller's borrowed array. I retain both
complete use-after-free reports, the original executable and source snapshots.
Both unchanged probes pass with fresh candidate providers.

My new permanent C/API test exercises shared and mutable callable/string
views, same-value/replacement binding, nested array/record/tuple reachability,
caller survival and exactly-once owning-root reclamation. It passes normally
and with fresh complete ASan/UBSan providers and leak detection enabled. The
first instrumented build and run finishes in65.406s. My unchanged nine-route,
64-call audit retains zero callable names, including with leak detection enabled.

My complete original134-method interpreter and88-method typechecker suites,
environment symbol-index controls, callable-expression and escaped-record
fixtures pass normally and under ASan/UBSan with leak detection disabled for
those complete fixtures. This is not whole-fixture leak acceptance. I retain
the earlier572-byte/26-allocation expression and3720-byte/159-allocation
record/union failures in the linked prior archive; general #1116/#1140
reclamation remains open. I preserve two diagnostic checker link refusals:
my first observer incorrectly linked evaluator test-hook objects to the
standalone checker. The corrected observer uses its ordinary provider objects;
both normal and instrumented full checker runs pass.

My source checker still restricts explicit borrows to nominal record owners at
declared direct-call arguments. The callable/string view tests exercise the
existing C/API ownership boundary and do not admit those source borrows.
All producer/consumer tools and Puck-owned source bytes remain guarded;
`nvm2c.c` remains SHA256 `994d83e225343cfee19f5674cc30259fcea6bc3db44d65911764fd63489f31c0`.
I claim no Darwin/current-head bootstrap or whole-release acceptance.

I verify my lossless archive with:

```sh
python3 docs/evidence/borrowed-value-ownership-linux-20261123/read_archive.py
```

My reader verifies1489 records,1385 exact Git inputs, source/provider guards,
compressed/uncompressed hashes, full retained failure streams, result matrices
and all actual normal/instrumented test products. My manifest reuses identical
objects from the earlier callable-expression and scalar-copy archives. Their
manifests are hash-bound; my reader independently verifies every referenced
object. The source/test/observer bytes, commands and original private paths
remain in the archive. I preserve all old roots, dirty audits and stashes, and
perform no NAS/TCC/migration operation.
