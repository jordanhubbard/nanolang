# AOT byte-character conversion

I adapt exact builtin `vm_string_from_char` and `string_from_char` imports:
one integer parameter and a string result. I preserve the native C `char`
conversion, not Unicode scalar encoding. A converted zero byte produces empty
C text. Each result has independent allocated storage, currently retained until
process exit. Conversion outside the C character range follows the host C
implementation, as in the existing native helper.

The extreme-input tests also exposed an AOT integer-literal bug. I now emit
`INT64_MIN` as `(-9223372036854775807LL - 1LL)`, avoiding an out-of-range
positive literal before unary negation. I retain strict C warnings.

## Verification

`make -j1 test-nvm2c` passes 904 checks on Darwin. `git diff --check` passes.
`make -j1 test-one-ir-compiler` remains failing after 2.242 seconds at the import
reported below.

Generated executables exercise both spellings with 0, 65, 127, 128, 255, 256,
257, -1, `INT64_MIN` and `INT64_MAX`. Each preserves its result after a second
conversion, and each rejects a non-integer input tag. A separate arithmetic
case checks that the signed minimum plus maximum is exactly -1.

Compiler preflight advances to import 34 (`vm_mktemp_dir`). Full compiler
acceptance remains open under MAC `task_419c47bdc8fc42e4b52eb6af1a0e9a71`.
