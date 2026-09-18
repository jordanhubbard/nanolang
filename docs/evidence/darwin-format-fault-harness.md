# My Darwin formatting fault-injection harness

I qualify only the private `snprintf` fault injector in
`tests/nanovm/test_substring_contract.c` under
`task_8674e6954ba44d3da54ea5809d0bba9c`.

The originating aggregate-policy gate reached a strict Darwin compilation
failure because the active SDK had already defined a fortified `snprintf`
macro. Root retained that first log as
`/tmp/nanolang-aggregate-policy-darwin-adjacent-first.log`. I did not replay
the failed artifact or classify this as a product defect.

My roadmap contract is commit `0639afed`. My production checkpoint is
`173b202b`, based on canonical `79b7177f`. The correction adds a guarded
`#undef snprintf` immediately before the translation unit installs its
existing `format_snprintf` mapping. It does not change production source,
compiler flags, SDK fortification, fault cases, assertions or deadlines.

On Darwin arm64 with Apple Clang 21.0.0:

- `make -j8 test-vm-substring-contract` passes. I retain
  `/private/tmp/nanolang-darwin-format-substring.log`, SHA256
  `0bdf9a1a94b51a2175e1ccd9a41d3806cf0cb4111f50e2a3ea5ca51358a82eda`.
- `make -j8 test-nanovm` passes the substring and callback-allocation
  companions plus 274,541 NanoVM checks with zero failures. I retain
  `/private/tmp/nanolang-darwin-format-nanovm.log`, SHA256
  `be3ead1d8f9706e221e1848d4c2f737cc82240aef7dfb4f60d0f04b811ec78f4`.

In a clean detached Linux arm64 worktree at the same source, the disposable
`python:3.12.12-bookworm` image has SHA256
`c0abd0758831ad99b7a29e0c1a875da9c4abb9a2e3f21e2eeb585dbcadfb6cd0`.
After installing the ordinary Debian build prerequisites inside that
container, `make -j8 test-nanovm` passes the same substring and allocation
companions and all 274,541 NanoVM checks with LeakSanitizer enabled. I retain
`/private/tmp/nanolang-linux-format-nanovm.log`, SHA256
`94f55066a625c02d597c0b7ed60364b0bc6e98191d27e4d2f6f9931e15782d1a`.

This evidence closes the test-harness prerequisite only. It does not qualify
aggregate binary64 semantics, product PR522 or release publication.
