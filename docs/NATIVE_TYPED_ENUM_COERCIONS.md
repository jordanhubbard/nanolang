# My typed integer enum coercion contract

I track `task_a77ca354773245f9a8b5f490fe336c06` on main `e8d18b20`.
My VM admits enum ordinals alongside int operands only for these typed binary
operations: I64_ADD, I64_SUB, I64_MUL, I64_DIV_S, I64_REM_S, I64_EQ,
I64_NE, I64_LT_S, I64_LE_S, I64_GT_S and I64_GE_S.
Arithmetic returns int and comparisons return bool. Integer wrapping and total
division/remainder keep their existing definitions.

I normalize boxed int/enum operands to temporary integer slots immediately
before those operations. I check actual runtime tags; I do not change stored
producer values or globally relax nvalue_require_int. I64_NEG, generic MOD/NEG,
float operations, integer function returns and other exact integer consumers
retain their existing guards. BOOL/U8/float/void and heap are not implicit ints.

I require paired VM/native values and tags for both operand orders, all eleven
operations, zero/wrap boundaries, shared locals/calls and retained producer tags.
Invalid-tag controls must stop at the expected guard without sanitizer findings.
I run GCC/Clang sanitizer controls and adjacent scalar/integer gates. LLVM/Wasm
must add operation-specific coercion with its separately recorded enum lane;
I do not edit that lane or claim the parent complete from native evidence alone.

My native source checkpoint is `8fe01034`. Seven GCC methods pass in 30.779
seconds: six new typed-enum methods and the existing total-integer boundary
gate. Thirty Clang methods pass in 30.449 seconds: the new controls plus enum,
numeric-union, tagged arithmetic and U8 tail-return controls. Generated native
programs use ASan/UBSan/LSan; the existing total-integer gate uses UBSan at O0/O2.

I retain my initial test failure at `/tmp/nanolang-typed-enum-gcc.log`: the
assembler correctly refused a statically known enum operand to F64_ADD before
runtime. I preserve that exact verifier/output boundary in its own test and use
a global-boxed operand for the separate runtime guard check. I do not change
production to make an invalid module admissible. Passing logs are
`/tmp/nanolang-typed-enum-gcc-final.log` and `clang.log` under the same prefix.
The full native gate remains pending while this checkpoint is draft.

This is the native part of a77ca. Its original cross-backend obligation remains
open until the dependent LLVM/Wasm enum lane supplies its own implementation
and acceptance. Parent66a6 remains open as well.
