# My NanoCore export buffer checks

I check buffer sizing, allocation and formatting before publishing an export.
A failed growth retains the old allocation until final cleanup. I latch failure,
stop appending, and return NULL through my existing public boundary. Successful
output retains caller ownership. I do not change reference evaluator transport
in this repair.

My production change is `8804c8df`; fixture linkage is corrected in `7acea7be`.
I pass ordinary 2047-byte growth and formatted output, plus deterministic initial
allocation, growth allocation, size overflow, first-format and second-format
failure controls under GCC and Clang at strict O1 with ASan, UBSan and leak
checking. Logs: `/tmp/nanolang-exporter-buffer-gcc.log` and
`/tmp/nanolang-exporter-buffer-clang.log`.

My original GCC13 strict O1 diagnostic is retained in
`/tmp/nanolang-exporter-format-baseline.log`. The corrected production compile
passes without warning suppression in
`/tmp/nanolang-exporter-format-corrected.log`.

My first full suite attempt lacked the fixture's standard argument globals;
I preserve that link failure in `/tmp/nanolang-exporter-nanocore-gate.log`.
After adding those fixture globals, `make -j8 test-nanocore` passes in
`/tmp/nanolang-exporter-nanocore-corrected.log`. I do not count the failed setup
as a passing gate or a production defect.

These checks complete tasks `task_e926ca38a5d64f299e9532ed984a2860` and
`task_927d53891d204f2fb4e1974eb8c3edc2`. They do not establish full release readiness.
