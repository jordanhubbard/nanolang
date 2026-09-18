# I lower my admitted owned-profile binary64 scalars

I record `task_cb823e6c37e6465e8d6ad10f3b726364` before production under the
mixed ordinary/owned proposal `task_4be28fef163f42069064357639b3b5cc`.

My owned verifier already admits PUSH_F64, float local copies, F64_ADD/SUB/MUL/DIV,
F64_NEG, six F64 comparisons, and generic EQ/NE/LT/LE/GT/GE with two matching scalar
tags. My native owned emitter lacks F64 cases and generic comparisons currently
read integer storage. I repair only that native lowering gap. I keep existing
source, parameter/result, resource-field, reference and managed-value admission.

I add a distinct binary64 scalar carrier member and explicit runtime tag to the
private generated value. Every literal/result producer establishes its actual tag;
LOAD/STORE/DUP/SWAP, owner field transport, reference access and calls preserve it.
A borrowed scalar call result establishes the callee's already-checked result tag.
I do not guess a float tag from stack depth or integer storage. Generic comparison
chooses scalar semantics from these tags after existing exact verifier admission.

I transport literal bits with memcpy from the decoded uint64 representation,
without decimal formatting or signed integer casts. Copies preserve those bits.
F64 binary arithmetic uses my existing shared nano_rt_f64_* provider: rounded
binary64 results, canonical arithmetic NaNs and either signed-zero divisor yielding
positive zero. F64_NEG follows the VM unary operation. F64 predicates use direct
IEEE comparison: NaN is unequal and all ordered predicates are false.

My generic float equality uses C floating equality, as vm/value.c::val_equal does.
Generic ordering uses the VM three-way comparison: a<b gives -1, a>b gives 1,
otherwise 0. Consequently NaN generic LE/GE are true, whereas F64_LE/F64_GE are
false. I preserve that existing distinction, including signed-zero equality.
Mixed INT/FLOAT generic operands remain rejected by affine_bytecode's matching-tag
rule; I do not silently broaden it to the VM's ordinary mixed-tag behavior.

My first production checkpoint precedes execution and receives independent review.
After review I require ordinary newly generated valid modules containing actual
owner operations plus float scalar work, VM/native matching results, every generic
and F64 comparison across signed zeros, finite values, infinities and NaNs, and
arithmetic/transport bit observations through the generated private carrier where
public float returns remain refused. I include local/stack copies, branches, loops,
owner field observation where already admitted, helper scalar results and existing
integer/bool/u8 controls. No unsupported module is executed.

I run strict GCC and Clang compilation, applicable sanitizer/leak controls and the
existing owned graph/result/reference regressions. I preserve first terminal
outcomes and source/tool identities. No Samples/Bundle/source-family completion or
new managed runtime/LLVM/Wasm admission follows from this prerequisite.
