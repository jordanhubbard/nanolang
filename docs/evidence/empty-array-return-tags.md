# My empty array return tags

I traced the native `narr_t`/`nsarr_t` result conflict in `extract_type_args`
to bytecode lowering. Its declaration returns `array<string>`, but its direct
`return []` emitted `ARR_LITERAL TAG_INT 0`. The nonempty path returned a
correctly tagged string array. My native translator correctly rejected those
incompatible representations.

I now retain the current function's declared array element type in my lowering
context and apply it to direct empty return literals. Nested function lowering
saves and restores that context. I do not reinterpret explicitly tagged
integer-array bytecode as string-array bytecode, and I do not relax native
result compatibility.

## Verification

- `make -j1 test-nanovirt nano_virt`: 74 codegen tests pass. The new matrix
  executes empty and nonempty return paths and inspects emitted empty-array
  tags for int, float, bool, string and record results. A nested integer-array
  function checks restoration of its enclosing function's return context.
- `python3 -m unittest tests.test_one_ir_compiler.OneIrCompiler.test_declared_empty_array_returns_reach_native`:
  int, string and record fixtures pass through source compilation, shadows,
  bytecode, native C emission, warnings-as-errors C compilation and execution.
- `make -j1 test-one-ir-compiler`: the focused fixture passes; the full compiler
  gate remains failing. Fresh bytecode gets past the earlier result conflict
  and stops at function 463 because my native translator permits only 256 locals.

I have not established contextual typing for every expression containing an
empty array. This change covers direct return literals, not a new general
inference rule. Full compiler acceptance and the release remain open.
