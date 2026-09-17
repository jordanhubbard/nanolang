# Tagged scalar-local convergence

I retain tagged storage when a local receives a runtime-tagged value, even if
another assignment supplies a concrete integer, boolean, or string. I record the
storage requirement during fixed-point inference and revisit earlier writes.
Reassigned parameters carry the same requirement back to their caller ABI.

I reuse my checked boxing and scalar extraction. Known incompatible payloads
still fail shape validation; unknown payloads retain their tags until an executed
consumer checks them. An untaken branch does not consume its invalid value.

Validation on integrated main `83aa198c` plus this change:

- `make test-nvm2c`: 2,215 native checks and 1,092 shape checks pass.
- Twenty-two new cases cover both assignment orders, ordinary locals, reassigned
  call parameters, three scalar kinds, taken/untaken invalid-value paths, and
  two exact incompatible-payload refusals.
- Twenty executable cases agree between VM and strict native C: sixteen succeed,
  four reject invalid consumption.
- All sixteen valid programs pass ASan and UBSan with default runtime checks.
- The unchanged `OneIrCompiler.test_compiler_bytecode_to_native_to_program`
  passes from a fresh build in 81.319 seconds. I compile my compiler to bytecode,
  translate it to structured C, compile that C under strict warnings, run its
  help command, use it to compile `nl_hello.nano`, and check the resulting output.

The last check establishes this compiler-bytecode/native execution bridge. It
does not establish the NanoISA-only bootstrap fixed point, emitter completeness,
resource correctness, or full-roadmap release readiness. Recorded returned-value
cleanup defects also remain separate.

Local logs: `/tmp/nanolang-native-tagged-locals/final-gate.log`,
`integrated-fullcompiler.log`, `parity.log`, and `sanitized.log` in that same
directory. The earlier fresh build also passed the unchanged compiler gate in
83.140 seconds (`fullcompiler.log`).
