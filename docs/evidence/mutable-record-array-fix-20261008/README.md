# My shared record-array parameter correction

I keep forward storage conversion for caller-to-callee fields and add a
reverse alias-view constraint for the shared mutable handle. A tagged field
can inform an existing exact caller view without changing its constructor
kind; I require compatible payload facts and retain the emitted runtime tag
checks at projection. Unknown nested fields acquire the callee's write facts.
My former whole-array equality experiment remains archived separately.

I built my translator with separate BIN_DIR and OBJ_DIR under
`/private/tmp/nanolang-diag-tools` to preserve tools used by the live source
snapshot suite. `make -j2 test-nvm2c` with those directories and
`NVM2C_TEST_BINARY=/private/tmp/nanolang-diag-tools/test_nvm2c` exits 0:
2,431 native checks and 379 callable checks pass. I then add and run late nested
write and compatible/incompatible tagged-view controls: 2,605 shape checks
pass. The original native gate ran the preceding 2,553 shape checks.

My source matrix uses the C-seed bytecode producer and installed Stage2 from
clean bootstrap pin 17347f8b4. Direct, helper, forwarded and indirect mutation
through an alias, and the unchanged populated diagnostic formatter all pass
in VM and strict native C with ASan/UBSan/LSan. Two methods cover ten products.
I retain the invocation's environment in the following command:

```sh
PATH=/private/tmp/nanolang-cutover-llvm-tools:/opt/homebrew/opt/llvm/bin:$PATH NANOLANG_SELFHOST_COMPILER=/private/tmp/nanolang-capture-ae92c0488-cow/bin/nanoc_stage2 NANOLANG_TEST_NVM2C=/private/tmp/nanolang-diag-tools/bin/nvm2c python3 -m unittest -v tests.test_native_mutable_record_arrays
```

I include these controls in both fresh compiler-product routes. Their clean
qualification, canonical integration, Linux qualification, and full release
acceptance remain open. This correction does not establish all aggregate
metadata or backend obligations.
