# My retained GNU include-state repair report

I reproduce lost original source identity when preprocessor line/directory markers and an assembler include meet in the retained GNU source. My full witness loses `payload.S` in decoded lines and the compilation-unit name; three neighboring witnesses preserve it. My one-condition candidate restores the full witness while preserving its neighbors.

I restrict the logical-line adjustment in `input_scrub_pop` to source-buffer expansion. Ordinary included-file restoration keeps its saved logical line. My [standalone patch](include-state.patch) applies with zero fuzz to before-source SHA-256 `40e7e14b5845e3f48b3be1cde0ea0b073f3e787864bdeb9d06dea9b545acfa07` and produces exactly the native-qualified after-source `6ffe0ed82d02c8f92198b4c2b5200a8093b7eea0a07a96bf951eecabf021b036`. My [patch control](patch-control.json) retains the command/result and exact patch hash; both streams are retained beside it.

I use the exact retained source archive `binutils-2.46.0.tar.xz`, SHA-256 `d75a94f4d73e7a4086f7513e67e439e8fcdcbb726ffe63f4661744e6256b2cf2`. The resulting assembler reports `2.46.0.20260210`. I do not infer compatibility for every distro build from that version spelling.

| Native Linux comparison | Baseline and candidate GAS results | Outcome comparison | Fresh witnesses |
| --- | --- | --- | --- |
| [AArch64 archive](../gas246-regression-20261121/README.md) | 1239 PASS, 10 UNSUPPORTED each | All 1249 records equal after only declared source-root mapping | All 12 expectations pass, including the existing GNU 2.42 reference |
| [x86_64 archive](../gas246-identified-x86-20261121/README.md) | 2091 PASS, 4 UNSUPPORTED each | All 2095 records equal exactly | All 8 baseline/fixed expectations pass |

I retain complete native GAS logs, original/fixed images, exact source/package inputs and raw/compressed hash audits in those archives. Their initial setup refusals and later successful observations remain separate. These are native assembler component results, not a checked theorem or all-target qualification.

I keep the exact [witness inputs](../../../tests/fixtures/as246-source-identity) available. From that directory, I compare each chosen native assembler using the same original stream and a fresh output directory:

```sh
capture_witness="$(mktemp -d)"
/path/to/as --gdwarf-5 -o "$capture_witness/witness.o" actual-markers.s
/path/to/readelf --debug-dump=decodedline "$capture_witness/witness.o"
/path/to/readelf --debug-dump=info "$capture_witness/witness.o"
```

The remaining three `.s` streams are neighboring controls; `debug macro.s` supplies their unchanged include. The full native recipe and all actual terminals are in the linked manifests.

I retain this report for review and submit nothing upstream. My separately identified private fixed profile and complete NanoLang source/response/cold/guide gates remain distinct from generic GNU 2.46 admission. I keep the known broken generic token refused.
