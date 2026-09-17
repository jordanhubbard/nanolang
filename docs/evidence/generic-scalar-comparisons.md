# My generic scalar comparison compatibility contract

I track task_01d3e7ca2b6a47b0bef9504f2fd04006. I preserve the existing NanoISA
VM contract in `val_equal` and `val_compare`; I do not extend my source-language
promotion rules or change typed arithmetic. My source comparison table permits
same-type equality and int/int or float/float ordering. Raw bytecode admits
additional tagged combinations, whose existing behavior I retain here.

For scalar tags void=0, int=1, u8=2, float=3 and bool=4:

- EQ compares equal tags by value; void equals void. Float equality is IEEE
  equality, including signed-zero equality and NaN inequality. NE negates EQ.
- Mixed int/float first converts the integer to binary64, then compares.
  This retains rounding: integer 9007199254740993 compares equal to float
  9007199254740992, and INT64_MAX compares equal to float 2^63. I do not replace
  this conversion with a mathematically exact mixed integer/float comparison.
- Other unequal scalar tags compare unequal. Ordering uses their tag numbers,
  not numeric promotion: byte zero orders after integer 100, and bool false
  orders after every float, including NaN.
- Same-type numeric ordering is numeric. For float/float or mixed int/float,
  the three-way comparator returns zero if neither `<` nor `>` is true.
  Consequently generic NaN LT/GT are false and LE/GE are true, while EQ remains
  false and NE true. This existing distinction is already documented in my
  native typed-float evidence. Typed F64 comparisons keep IEEE unordered rules:
  EQ/LT/LE/GT/GE false, NE true for NaN.

I use separate equality and ordering helpers. I admit the six generic
comparison opcodes only within the existing closed scalar LLVM/Wasm profile.
My C repair covers its missing concrete bool/float equality cases; existing
boxed and ordered comparisons remain unchanged. I preserve heap and string
comparison behavior and LLVM/Wasm refusal. Generic arithmetic has a separate
continuation and remains outside this admission.

My gate uses the same verified modules across VM, C, LLVM interpreter,
optimized/native LLVM, Wasmtime and import-free Node. I test all 25 ordered
tag pairs, integer rounding boundaries, signed zero, infinities, NaNs, boolean
result tags, calls/locals/joins, and eager operand evaluation. Unsupported
profiles must preserve previous output.
