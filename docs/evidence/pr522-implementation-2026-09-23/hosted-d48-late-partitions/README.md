# My later d48 hosted terminals

I retain the failed-job logs from run `35974675846` without cancelling the
remaining diagnostic partitions.

- Job `107552245336` repeats the ten native nominal-order failures on Linux ARM64.
- Units-05, `107552438145`, refuses `cast_bool` on both native stages before
  the u8 positive cases and expected shadow failure. I track
  `task_32f2fc1416e14f1f8f0740ac6fc0c3e6`.
- Units-06, `107552438048`, exposes inherited product-link configuration in
  three fixtures. My [local fixture correction](../product-link-fixture-environment/)
  preserves the product contract; final hosted acceptance remains open.
- Units-02, `107552438210`, repeats the 180-second Stage 2 borrow-parser compile
  timeout. Coverage job `107552245153` reaches its existing 2,700-second alarm
  while enum-metadata tests are active. I track both under
  `task_4251e719e9634aa7b597dbdd080b6b9b`. The coverage terminal does not establish
  that the active test caused the cumulative overrun.

These terminals predate the latest local fixes. A later corrected gate must
pass; neither a local pass nor the age of a failure establishes its cause.
