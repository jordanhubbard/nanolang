# My tagged scalar return boundary

I check a returned value's runtime tag against its declared function result
in NanoVM (`result_tag_matches` and `OP_RET` in `src/nanovm/vm.c`). Missing
map lookups produce void, but returning that void from an integer, string or
boolean function is a type error. I must not reject all optional values during
native shape inference or silently turn a missing result into zero.

I now keep tagged source storage separate from the declared native scalar
result shape. My emitted return checks and unwraps the value. I add boolean
checked unboxing alongside integer and string checks. Ordinary and tail callers
receive the declared scalar only after that check succeeds.

My test matrix exercises integer, string and boolean declared results, missing
lookups, present integer/string lookups, local storage, an alternate ordinary
scalar return, and ordinary calls followed by a tail call. Matching lookup tags
return their values; missing and wrong tags terminate the generated process.
Boolean lookup storage itself remains unsupported; these tests cover ordinary
boolean results and rejection of void/integer/string as a boolean result.

I verified:

- `make -j1 test-nvm2c`: 1,226 AOT checks and 994 shape checks pass.
- `make -j1 test-nvm2c-sanitizers`: the same checks pass with fresh verified
  ASan/UBSan translator and shape objects. I do not claim leak freedom.
- `python3 tests/test_nvm2c_opcode_coverage.py`: one check passes.
- `PYTHONPATH=. python3 tests/test_nvm2c_sanitizer_driver.py`: three checks pass.
- `make -j1 test-one-ir-compiler`: still fails at unsupported `LOAD_GLOBAL`
  in function 323 at offset 21. This change does not implement global storage.

I retain untyped results, general mixed-value joins, tagged global arrays and
full compiler acceptance as unfinished work. My AOT parent task remains
`task_419c47bdc8fc42e4b52eb6af1a0e9a71`; this evidence does not close it.
