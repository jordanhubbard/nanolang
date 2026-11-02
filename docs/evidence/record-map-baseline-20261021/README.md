# I retain ordinary map-field baselines

I ran the unchanged sources at `878c2bb83` before proposing map authority. `receipt.json` retains exact commands, source hashes, eight before/after provider hashes and individual terminals; `run.py` is the actual serial harness. I changed no installed provider or source fixture.

| Source | Seed | Fresh C-hosted Nano | Retained Stage1 / Stage2 |
|---|---|---|---|
| `tests/nanoisa/fixtures/map_record_fields.nano` STRING→INT | pass | pass | pass / pass |
| `tests/unit/test_map_field_operations.nano` STRING→STRING | pass | undefined `map_remove` | undefined `map_remove` / undefined `map_remove` |

For each accepted module I ran original verification, VM execution, canonical dump, nvm2c translation and strict O1 ASan/UBSan native compilation/execution. All accepted cases lack optional layout/ownership authority. I do not infer paired STRING→STRING support from its type classifier or the seed success. I did not produce separate selected-shadow modules in this baseline.

The C-hosted emitter is the prior isolated self-tail build at `/private/tmp/nanolang-self-tail-proof-20261021/emitter`; its Nano source and changed proof-source pins match this aggregate. Installed stages are retained generations, not newly generated from this baseline. All eight provider hashes remained unchanged. Native build budget was 900 seconds through the existing override-aware helper; guest commands retained 180 seconds.

This receipt establishes an authority omission for the paired STRING→INT anchor, not executable map authority or a new wire grammar. #1001 remains open.
