# My C-seed union callback signatures

I resolve a declared union's kind before comparing retained callback annotations
(MAC `task_f00d97409c26413781ff85f06993e66d`). My parser initially represents
an ordinary named annotation as `TYPE_STRUCT`. My function-registration pass
already resolves a named union parameter to `TYPE_UNION`, but my callback
signature previously retained the placeholder. This made an ordinary
`fn(Choice) -> int` disagree with the declared `read_choice(Choice)` function.
The earlier observation remains in [my constructor-context evidence](selfhost-constructor-call-context.md).

I now resolve exact declared union names in annotation trees and callback
parameter/result slots during nominal binding. I visit retained signature
annotations recursively, preserve nominal names and concrete generic arguments,
and exclude in-scope formal parameters from declaration resolution. I do not
relax signature equality or ownership checks.

My acceptance at production commit `6fc07b7ae7592606b63b8a993b4a32e339a8af20`
uses fresh normal compiler inputs:

- `python3 -m unittest -v tests.test_cseed_union_signatures`: nine methods pass
  in 7.546 seconds with GCC and 7.289 seconds with strict Clang. I execute local
  and computed calls, empty and stored union values, callback parameters and
  union results. I reject wrong nominal signatures, wrong union arguments,
  wrong payloads and unequal concrete generic arguments. A failed shadow
  preserves the previous artifact.
- `make -j4 bootstrap test-parser test-typechecker`: my fresh bootstrap,
  parser units and typechecker units pass.
- `python3 -m unittest -v tests.test_module_signature_metadata tests.test_parameter_nominal_metadata`:
  six methods pass in 8.233 seconds, retaining compiled-module callback metadata
  and distinct record identities.
- The existing C-seed `test_function_variable_and_computed` and
  `test_function_value_payload_refusals` constructor controls pass in 1.407 seconds.

- `python3 -m unittest -v tests.test_generic_function_values tests.test_resource_callback_boundary`:
  all 31 methods pass in 97.939 seconds across C-seed, Stage1 and Stage2, with
  the NanoVirt/VM control built and executed.

I restacked onto main `c1f18abcf8180819c0ba2e8d3ed08039e55595bc`,
preserving both additive Makefile targets. My production source and test file
are identical to the tested checkpoint above. The integrated
`make -j4 test-cseed-union-signatures` passes all nine methods in 7.526 seconds.

I retain local logs under `/tmp/nanolang-cseed-union-signature-*.log`. My first
build command used the nonexistent `compiler` target; no compiler ran in that
attempt. The supported `make -j4 bin/nanoc_c` build passed. I do not treat the
initial command error as a product defect.

My imported-module union ownership task
`task_a2f464df8ba84a4ab4fc52c509e96904` remains separate and open. These callback
tests do not establish that broader imported-union contract or resolve the
unrelated product metadata acceptance task `dd74`.
