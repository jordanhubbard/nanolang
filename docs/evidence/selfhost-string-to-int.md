# My string-to-integer conversion

I lower `string_to_int` to the existing `CAST_INT` instruction, infer its integer
result for enclosing arithmetic, and use a normal return after the conversion.
I require exactly one string argument. I do not change conversion runtime
semantics.

On Linux ARM64, `make -j8 test-nanoisa-src-nano` passes 86 baseline comparisons
and 15 focused Python cases. Eight new module/function checks match the C seed;
both emitted modules verify and execute positive, negative and zero values,
conversion inside arithmetic, and an integer-to-string round trip in NanoVM
and strict native C11. Four wrong argument/type cases fail without output.

Real `src_nano/nanoc_v06.nano` emission now first refuses `unsupported result
type array<string>`. I recorded that continuation as `task_4ea96b68ae4b43f7a0cfc16cd7c19649`.
This conversion slice is tracked by `task_62d9f8ab389e4299b1b16a33ed591630`.
Full compiler emission and bootstrap equality remain open.
