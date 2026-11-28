# My retained shadow timing comparison

I compare existing traces for #1229/#1241. I start **zero new native attempts**.
This is localization evidence; it does not renew failed worker admission or
qualify the current source.

```sh
python3 docs/evidence/source-shadow-timing-linux-20261124/read_comparison.py
```

My independent comparison checks the compressed/decompressed hashes and sizes of
six retained qualification records and six actual stderr traces from the earlier
[enum checkpoint](../native-enum-lists-linux-20261123/README.md) and
[helper-closure checkpoint](../public-c-fixture-recovery-linux-20261123/README.md).
I compare exact source/name/line event order, zero completed assertion failures,
unchanged 600/default10 limits and all 233 NanoLang source-input identities. Each
failed completed sequence is exactly a prefix of its mode's passing sequence.

| Failed observation | Completed prefix | Passing prefix elapsed | Failed prefix elapsed | Fraction of failed dispatch-to-status wall time |
| --- | ---: | ---: | ---: | ---: |
| Ordinary, unbound | 1294 | 5.173909s | 9.912262s | 94.28% |
| Ordinary, CPU7 admission | 1078 | 4.758013s | 9.929719s | 96.58% |
| Private, unbound | 1076 | 7.625425s | 9.297518s | 92.87% |
| Private, CPU7 admission | 1048 | 7.094670s | 9.903834s | 98.95% |

The passing full 1,306-method body-time sums are 5.338705s ordinary and 8.872598s
private. These sums measure each trace's completed-body wall durations. They are
not child CPU measurements and exclude runner overhead and each failed run's
unfinished final method. Dispatch-to-status can exceed 10s when the parent is
scheduled later; all four retained child statuses report SIGALRM and no completion
receipt. I do not interpret that reporting gap as extra admitted execution time.

The comparison shows broadly longer completed-body observations; it does not
support locating these four failures solely after the final shadow or in receipt
flushing/reaping. It identifies no host, CPU-class, compiler, collector or product
cause. The first #1229 incident with all 1,306 completions remains a distinct
retained terminal; these four traces do not retroactively diagnose it.

I retain every original failure, source and image. A bounded future performance
localization must preserve assertions, source identities and original deadlines.
I authorize no unchanged full-run retry through this comparison. Full-source,
paired-platform, raw/installed/default-ownership and all 69 release requirements
remain open. No Darwin job, host policy or NAS/TCC/migration action is involved.
