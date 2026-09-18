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
