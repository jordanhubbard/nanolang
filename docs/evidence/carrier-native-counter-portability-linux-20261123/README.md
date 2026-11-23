# I keep my native-byte test linkable without counter support

I qualify test-only source `04ab8e3bedc00cd2eabbb7f91074b34fc95407fd`, following my [private collection checkpoint](../carrier-collection-scratch-linux-20261123/README.md) at product source `ff8b95a1c9601fd94decd2b7500676314b0b1230`. Product source, NanoLang modules and Makefile remain unchanged. My reader verifies246 records and the precise fixture-only edit.

My original test put all included evaluator helpers beneath the counter-support guard. Its intended unsupported-host skip then failed linking clock/metadata/I/O stubs required by the existing test objects. An external Linux control forces only those counter-selection branches and reproduces the undefined references. I retain its exact source, link arguments and streams.

I now compile shared fixture helpers on both branches and select the skip inside main. The forced no-counter control links and executes its exact skip message. Normal and ASan/UBSan+LSan controls execute the full original128-call/ten-shadow workload, keep the1024-byte threshold and2MiB positive controls, and both report zero native-byte growth. I preserve every original measurement-path statement/assertion; the standalone reader reverses only the support guard movement and relative include context to recover the prior fixture exactly.

This is an actual Linux branch/link control, **not native Darwin execution**. An unavailable counter remains an explicit skip; it does not become accepted heap evidence. The preceding product qualification, historical failures, five frozen tools, default ownership and all remaining platform/raw/release requirements remain unchanged. All native jobs are terminal; no Darwin reservation or NAS action.

```sh
python3 docs/evidence/carrier-native-counter-portability-linux-20261123/read_archive.py
```
