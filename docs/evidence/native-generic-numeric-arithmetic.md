# My bounded native generic numeric arithmetic

I separate generic known-numeric promotion from typed I64 lowering. My
classifier retains a float result for ADD/SUB/MUL/DIV if either known operand
is float, and for NEG of a known float. My emitter requires both promoted
operands to have known int/float storage and converts an integer with the
ordinary binary64 cast before evaluating the operator. MOD remains integer-only.

I inspected the ordinary handlers in `src/nanovm/vm.c` before this change:

- Integer ADD/SUB/MUL and NEG wrap; integer division/remainder are total at
  zero and the minimum-integer/negative-one boundary. My existing unsigned
  native helpers and guarded integer routes remain unchanged.
- Float arithmetic and mixed int/float arithmetic return float. Conversion
  rounds to binary64, including integers beyond its exact range. NEG retains
  signed zero. NaN and infinities follow the ordinary floating operators.
- A float division by positive or negative zero returns positive zero even
  with a NaN or infinite numerator, matching the VM's explicit zero guard.
- Boolean, U8 and void are not numeric operands. The VM has additional
  enum coercions for four binary operators and string/array behavior; this
  child does not expand or establish those native contracts.

I retain the previous checked-int route for tagged operands. I do not claim
runtime float promotion when static inference lacks an exact numeric kind.
A known float paired with a tagged value is refused before output replacement;
a tagged value reaching an integer route must pass its exact integer check.
This is a bounded admission, not full generic arithmetic parity.

## My acceptance

`tests/test_native_generic_arithmetic.py` exercises the same retained modules
through NanoVM and standalone C compiled by GCC and strict Clang with
ASan/UBSan/LSan. It checks all four concrete int/float pairs for each promoted
binary operator, exact result tags, binary64 rounding, NaN/infinity, signed zero,
zero division, call/operand order, locals, integer overflow boundaries and
checked tagged integers. Invalid bool/U8/void and float MOD either refuse native
publication with previous output preserved or reach the existing exact-tag
guard. Guard tests require SIGABRT and no sanitizer report, not merely failure.

My initial combined GCC gate passed eight existing float methods and six new
methods in 69.521 seconds. After adding nonnumeric float-pair controls and
NaN/infinity zero-divisor cases, seven methods passed in 24.650 seconds.
The existing 38-case typed/generic wrapped-integer gate also passed.
The seven-method Clang gate passed in 3.158 seconds. The strengthened guard
method passed on GCC in 13.954 seconds and Clang in 2.132 seconds. My full
native suite passed 2,422 checks; the shape suite passed 1,092 checks.

The production change was first committed as `348b6911`; its additive restack
`e66d25a5` has identical source. I retained local logs under
`/tmp/nanolang-generic-arithmetic-`, including `focused.log`, `wrapped.log`,
`final-gcc.log`, `final-clang.log`, `guards-gcc.log`, `guards-clang.log` and
`native.log`. I ran no preserved failing compiler artifacts.

My child is `task_ec1b78703b0f49cbb4e05d8e8b7779df`. Parent
`task_66a6dd8ca51d415f9efb0f2904f85b49` remains open for broader tagged numeric
promotion and applicable backend parity. LLVM/Wasm work is separate; this
native child neither implements nor closes it.
