# My imported empty-array fixture

My comprehensive runner treats every `tests/unit/**/*.nano` file as a standalone
program. The empty-array return helper is an imported module with shadows and
no `main`, so it belongs in `tests/fixtures/`. I moved it unchanged and updated
the importing test's relative path. I added no runner exclusion or synthetic
entry point.

`python3 -m unittest tests.test_empty_record_array_fields` passes its native C
and NanoVM compilation, shadow execution and runtime checks, including the
imported boolean-array return.

The real `tests/run_all_tests.sh --unit` discovery now selects 30 standalone
programs and passes the empty-record-array case. A direct run passed 28 and
hit the two separately tracked live-MAC dependency-shadow timeouts. With the
existing `tests/fixtures/offline_mac` directory prepended to `PATH`, the same
runner passed all 30, with zero failures and zero skips. I did not change its
MAC policy in this fixture relocation.
