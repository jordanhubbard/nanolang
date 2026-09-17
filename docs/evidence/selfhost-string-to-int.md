# My string-to-integer lowering

I lower `string_to_int` to NanoISA `CAST_INT`. I infer an integer result, keep
the call inline in return positions, and require exactly one string argument.
Wrong arity, integer arguments, and boolean arguments fail without publishing
assembly.

On Darwin ARM64, `make -j8 test-nanoisa-src-nano` passes 86 baseline bytecode
checks and all 15 focused Python cases. The conversion fixture adds eight
named module/function checks. Both the C-seed and self-hosted modules verify
and execute decimal, negative, and invalid text in NanoVM and strict C11
`nvm2c` output. Four malformed calls are refused.

I reran real `src_nano/nanoc_v06.nano` emission. It now passes the
`string_to_int` boundary but still refuses a later form in the pinned subset.
That is progress through the compiler, not complete compiler emission or
bootstrap equality. This slice is tracked by
`task_62d9f8ab389e4299b1b16a33ed591630`.
