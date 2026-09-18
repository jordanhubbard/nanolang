# My native enum scalar contract

I track this child of66a6 as `task_29fc04d3d16e4caca9938ac2da4ebe0b`.
I transport ENUM_VAL with its actual tag9 and ordinal in my existing boxed
scalar carrier, never as a heap handle or an integer with its tag erased.
I preserve locals, calls, returns and generic tag checks.

My VM coerces enum operands to integers for ADD/SUB/MUL/DIV, then applies
ordinary int/float promotion and total integer arithmetic. MOD and NEG reject
enums. BOOL/U8/void and heap arithmetic keep their existing boundaries.

I preserve the current VM compatibility rules without broad semantic repair:
enum/int equality and ordering compare ordinals; enum/float equality is false
and ordering uses tag order; same-enum generic ordering returns zero.
CAST_INT returns the ordinal; CAST_FLOAT returns zero; truthiness tests the
nonzero ordinal. I audit display separately and do not pass enum payloads to
string or heap helpers. These compatibility rules are not new numeric claims.

I require ordinary paired VM/native GCC and Clang sanitizer tests for values,
actual tags, transport, both arithmetic operand orders and exact MOD/NEG
refusal. Invalid publication must preserve previous output. I retain existing
integer/numeric and native solver gates. I do not widen nominal heap layouts.
LLVM/Wasm task3762 is dependent and remains separate until its owner is free.

I keep typed integer enum coercion separate as taska77ca. My VM coerces
enums in selected typed binary instructions; native exact integer extraction
still rejects them. This child changes only generic arithmetic and preserves
that explicit native refusal. CAST_STRING keeps the existing empty fallback;
PRINT/PRINTLN use `enum(ordinal)` as my VM does.
