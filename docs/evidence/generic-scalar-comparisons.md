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

## My measured acceptance

My base is `045fb5a2` (merged scalar string allocation checks). Contract
`e74c0b85` precedes implementation `bfeb61b3`. I change no VM comparison code.

- My seven new methods pass in 6.986 seconds, exercising the same verified
  modules on VM, C, LLVM interpreter, optimized LLVM, linked LLVM, Wasmtime
  and import-free Node (`/tmp/nanolang-generic-comparison-r1.log`).
- They pass in 15.208 seconds with generated native C instrumented by
  ASan/UBSan/LSan (`/tmp/nanolang-generic-comparison-sanitized.log`), and in
  6.860 seconds with strict Clang (`/tmp/nanolang-generic-comparison-clang.log`).
- They also pass in 10.779 seconds with the host LLVM translator sources
  instrumented by ASan/UBSan/LSan; existing linked ISA objects and generated
  LLVM machine code are not fully instrumented
  (`/tmp/nanolang-generic-comparison-host-sanitized-final.log`). My first manual
  sanitizer build omitted the module include directory; I retain that setup
  error and the corrected command's build log separately.
- My existing native translator and shape gates pass 2,414 and 1,092 checks
  (`/tmp/nanolang-generic-comparison-native.log`).

The 55-method shared run produced 54 passes and one execution error in
125.674 seconds (`/tmp/nanolang-generic-comparison-common.log`). I had incorrectly
launched `make test-nvm2c` concurrently: it relinked `bin/nvm2c` while the first
method tried to execute that path, producing PermissionError. The retained
native build log shows the relink. I froze an identical executable copy
(hashes in `/tmp/nanolang-generic-comparison-frozen.sha256`) and reran only the
affected calls/loops/boolean method, which passed in 0.126 seconds
(`/tmp/nanolang-generic-comparison-affected.log`). I do not report the initial
combined invocation as a clean pass or classify an unexplained product failure
as infrastructure.

All 55 methods therefore have passing execution evidence, with that explicit
coordination correction. I did not rebuild the full compiler, change frozen
acceptance products, or claim complete LLVM/Wasm coverage. Generic arithmetic
continues under task_66a6dd8ca51d415f9efb0f2904f85b49.
