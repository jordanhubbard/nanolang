# Canonical recursive scalar arrays

I qualified recursive arrays whose leaves use my existing int, bool, float,
string or enum storage. I still refuse recursive record arrays. I retain the
declared element shape through empty literals, filled arrays, push and set;
the generated native carrier traces child arrays through my aggregate roots.

The release repair is `5175004e37e1e744ba4c3d57622d4910771ec0c6`.
My final changed-source hashes are:

```text
9ded100b2ff4caab4e0f90c1043a3abff66ef069000b8cee2150df62cf5c45cd  src/nanoisa/nvm2c.c
2cb37e8099d8f154bdf5c35952adec61e41cd36192d20670865d7be0bede4dcb  src/nanoisa/nvm2c_map_roots.inc
6c8d7ec2944e7e0f4defc8415dc43c089cc94440e78c7ca87db67980d7b8e8a4  src_nano/compiler/nanoisa_codegen.nano
94c04cf300182e27dbdabba44db611b39188c32406efecf49a263b1f6bd72b98  tests/nanoisa/test_nvm2c.c
c1f1104ba86db63d451dfbad331740927eb40d6babc327fc04328f0dcbc94597  tests/test_nanoisa_flat_records.py
dfaaea4a1b02d4efa4be4ad0d77b335622c7034a6b09862a98f6c327cd7afa6b  tests/test_selfhost_array_compatibility.py
b65fcca8e46e2cbfa6909e37e99d40dbd9442ecdac7e6d6afed97849f1e43ca2  tests/test_nanoisa_shadow_emitter.py
```

On Linux, a fresh detached checkout first passed the two-stage bootstrap and
installed compiler independence at compiler-source pin `25422283d`; the later
production change only split generated C statements for strict GCC warning
portability. The exact final head then rebuilt the affected tools and passed:

- all six unchanged `tests.test_selfhost_array_compatibility` methods in
  4.161 seconds;
- 1,412 shape checks and 2,426 native translator checks;
- all dependency shadows while rebuilding `nanoisa_emit`, 86 source-emitter
  comparisons, and all 90 adjacent VM/native emitter methods in 154.134
  seconds.

The retained log hashes are:

```text
6f67b416f8693ced9098550302303ea73fa9aab8a7e8df68dd9737285096b013  bootstrap
c7db09ec4c95669f262d96c01540c99a7a9ff983fee81767fca160dfb3ed6335  array compatibility
107f57086935153853201ee264cff6900faa16da56b0d74d15a1f02cedfb9641  native translator
77701494611fc7b0e798d19854ec3679744b9dfe7d9f80e5e6cb3ace75bf18dc  source emitter and shadows
```

The earlier exact-head hosted x64, arm and coverage failures all named the
same unsupported nested local. I did not classify those terminals as flaky or
replace this work with my C-seed route. Hosted release CI remains a separate
required gate.
