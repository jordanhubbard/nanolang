# I bootstrap retained service origins at 4bf302505

On Darwin, all seventeen steps pass through the C NanoISA seed and independent
self-hosted Stage1/Stage2. The raw Stage1/Stage2 modules are byte-identical. I
retain complete step logs and the source/tool/product manifest. Every source
hash in that manifest still matches the development tree at collection.
Stage1 takes 226.951 seconds and Stage2 takes 232.324 seconds.

I ran:

```sh
CC=/opt/homebrew/opt/llvm/bin/clang NANO_NATIVE_TEST_CC=/opt/homebrew/opt/llvm/bin/clang make bootstrap
python3 -m unittest -v tests.test_service_origins.ServiceOrigins.test_actual_drivers_reject_duplicate_service_origins_before_lowering
```

All four actual drivers (nanoc_c, nano_virt, installed Stage1 and Stage2) reject
a duplicate service origin in the new binding phase and preserve prior output.
The additional driver regression changes no compiler input from 4bf302505.
Earlier paired helper and recursive-loader coverage remains in
[my focused evidence](../service-origins-20261009/README.md).

This qualifies origin retention in the rebuilt driver. I have not connected
companion acquisition, complete namespace/nominal checking, either File lowerer,
selected service shadows, grants or staged source publication. Linux and the
complete #989/#976 acceptance remain open.
