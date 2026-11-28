# I bound private collection metadata without accepting default ownership

I qualify source `ff8b95a1c9601fd94decd2b7500676314b0b1230` on native Linux ARM64, following my [record-list compiler checkpoint](../record-list-loop-linux-20261123/README.md). My reader verifies **2,971 records / 1,426 exact current Git inputs / 58 distinct native ELF images**, including retained older headers and sources, actual provider objects, controls, failed candidates, native products and scripts/streams. Unchanged content-addressed objects may refer to earlier archives; their compressed and raw hashes remain independently checked.

```sh
python3 docs/evidence/carrier-collection-scratch-linux-20261123/read_archive.py
```

I reuse a session-owned collection root vector, grow it only with peak root demand, clear copied references after success or failure, refuse reentrant collection, preserve the previous vector on failed growth, and free it at teardown. This stores metadata, not retained payload ownership. I also close linear-probe holes at carrier removal, preserve entry/context identity through wrapped clusters, and revisit entries moved during sweeping. I avoid repeated same-capacity table allocation without adding a payload retention cache, changing an allocator setting or changing public layouts.

| Actual check | Result |
| --- | --- |
| Unchanged original native observer, older providers, scratch-only candidate | **Retained failure:** exit94, +1584 bytes; positive2101904, all128 calls/ten shadows and original1024-byte criterion intact |
| Same original observer/provider set, scratch + compaction | PASS, zero growth; only the two intended provider objects replaced; other original provider hashes unchanged |
| Permanent original workload, fresh normal providers | PASS, after64=158608 / after128=158608; positive2101904 |
| Permanent original workload, fresh ASan/UBSan providers | PASS with LSan, after64=205328 / after128=205328; positive2097152 |
| Scratch refusal/lifetime controls | Normal and ASan/UBSan+LSan pass failed growth, failed collection, reentrant refusal, repeated reuse, changing roots, borrowed caller survival and exactly-once metadata teardown |
| Registry controls | Wrapped collisions, forget/readoption, moved sweep entries and1024 changing-key rounds pass without another table allocation; old registry fails the unchanged new allocation check |
| Original engine/adapter targets | Normal and ASan/UBSan+LSan pass all original six allocation failures,65536-node graph and adapter refusal/rollback controls; their original sources/assertions remain unchanged |
| Original source/ownership neighbors | Normal and ASan/UBSan+LSan pass STRING scopes, active roots, STRING-only/full modes, record-list loops and checker metadata |
| Original evaluator/checker | All134 /88 cases pass normally and under ASan/UBSan; broad sanitizer runs explicitly disable LSan |
| Actual compiler driver, ordinary and explicit STRING candidate | Seven unchanged payload methods pass in each image, including original assertions, prior-output negatives, budgets and66/126-character names |
| Full original compiler source, ordinary image | PASS108.647s, all1306 dependency/root shadows, actual published compiler --help exit0 |
| Full original compiler source, explicit STRING image | PASS109.615s, all1306 shadows, actual published compiler --help exit0 |
| Policy/frozen tools | 3468 static fragments, zero reserved names; five original tool image/size/symlink guards unchanged |

Both full compiler-source stages keep the original600-second parent limit and default shadow deadline. All guarded sources, including main.c and the private metadata files, match before/after. Each native compiler product is26,105,208 bytes:

- Ordinary source-stage product SHA256 `4c9a7425e57ca22f4d8bd66ff06f2f2d0c20810e7364355046c761f698f20ad6`.
- Explicit STRING source-stage product SHA256 `0f77f28176d6fe880c8d667063623a78f0092ed051168eeee97ffbd8e3a3a9a6`.

I retain the diagnostic sequence. In one unchanged-observer run, three additional freed528-byte chunks match env_carrier_collect backtraces. The scratch-only candidate still fails; its read-only follow-up identifies five528-byte frees in carrier_resize(capacity32), with observed+1648 bytes. The two-part correction then passes the original observer under ordinary execution. I do not attribute the older+1776-byte observation to exact chunks I did not capture or infer a historical timeout/performance cause.

I also preserve the initial isolated link refusal: compiling the older runtime against the newer header omitted its old inline env_carrier_enabled helper. I restore the exact retained older headers, then rerun only that changed setup. My original public header differs only by the already-qualified record-list helper prototype; no layout changes enter this control. A later guard-script comparison initially compared digest-only values to structured guards; its retained digest map matches every corrected hash. No tool bytes changed.

My first scratch-only normal/sanitizer build logs remain, but their mutable unit/provider paths were rebuilt after compaction. I do not claim to retain all intermediate passing images. The separate failed isolated candidate, failed link object, final provider products and both complete source-stage products are retained. The reader verifies final products against their own observations rather than assigning final bytes to earlier passing logs.

I accept the original **private Linux native-byte criterion** for this workload. #1213's broader qualification, #1217 default STRING safety, #1211/#1116 aggregate ownership, raw/self-hosted stages, paired platform bootstrap and acceptance of the complete69-requirement release remain open. Ordinary environment creation and CLI ownership activation remain unchanged. Puck's sources/merge/capture/CI, frozen roots/audits/stashes and NAS/TCC/migration pauses remain untouched. Every native job in this receipt is terminal; I hold no Darwin reservation.
