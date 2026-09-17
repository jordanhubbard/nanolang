# My compiler AOT artifact-binding prerequisite

I reproduce this failure on unchanged integrated PR #383 head `5e078fb5`,
whose source was merged as `c7492089`, and again with generic PR #388. The
failure is independent of the generic classification changes.

```sh
python3 -m unittest -v tests.test_one_ir_compiler.OneIrCompiler.test_compiler_bytecode_to_native_to_program
```

The compiler emits its bytecode, but `nvm2c` rejects import zero,
`nl_nanoisa_load_print`, because the artifact-backed function lacks an exact
library binding and typed value adapter. The canonical `--emit-nvm` route
imports my NanoISA facade into the compiler. My translator's rejection remains
required until the host binding contract is implemented and tested.

I track the repair in MAC `task_600074c773904b119b39bdafd85c07a5`. Acceptance retains the existing
compiler-bytecode-to-native-to-program test, actual generated compiler execution,
typed facade behavior and rejection of unsupported artifact imports. I do not
count a removed import, disabled test or relaxed guard as a repair. This is a
required full v5.0.1 architecture gate, not a generic-classification pass.

Local baseline evidence is `/tmp/nanolang-main385-compiler-aot-baseline.log`
(5.821 seconds); the independent generic-tree reproduction is
`/tmp/nanolang-generic-affine-compiler-diagnostic.log` (5.965 seconds), on sparky.
