# I preserve scalar float locals across unsafe owner scopes

I record task_a490c997619140c78c6f150f40445062 before implementation, under installed-product task_e8d860a16da0464891dd32e91c42bef1 and full ownership task_28f2fb4b1f3c8a5ce93df628bb569d76. My base is canonical a2953371, including reviewed stack-float runtime PR798. This is independent of mixed managed-array descriptor work and transitive wrapper admission.

## My unchanged acceptance

I retain tests/test_owned_record_patterns.py::test_unsafe_pattern_keeps_outer_shadow and its full PREFIX. Main declares outer fd:float=2.5; an unsafe block destructures Handle{fd:42} and asserts its inner integer; afterward the outer float must still equal2.5. Original close/main shadows remain selected and executed. I do not rewrite the source or omit its resource/shadow declarations.

## My ordered boundaries

1. I admit exact nonparameter FLOAT locals in the public owned verifier. Inner affine scalar definition/load checks already include FLOAT and meet definite initialization over control-flow edges. My outer whitelist will permit FLOAT only at local indexes at or beyond function arity, with the existing no-layout/mode-zero declaration rules. Existing scalar parameters/results and resource-field lists remain unchanged. Native PR798 carries a distinct float and tag through full-value local copies; common VM locals already transport NanoValue. I add no format, heap or owner authority.
2. I keep scalar-local tag selection separate from resource leaves, borrowed/consuming parameters and results in both source producers. Float literals use existing decoded binary64 emission (C) and original float token spelling (Nano); canonical assembly must agree. Matching float arithmetic uses F64 operations and unary negation; comparisons use typed F64_EQ/NE/LT/LE/GT/GE, matching both ordinary source producers. Generic bytecode comparisons remain unchanged: their unordered three-way comparison behavior differs from typed F64 predicates. Mixed-tag arithmetic and implicit casts remain refused. I preserve once-only operand evaluation.
3. I lower checked unsafe bodies as ordinary lexical scopes containing only already-admitted operations. Unsafe does not grant foreign-call, pointer, array or hidden-owner admission. I reuse exact scope cleanup/name-end behavior, preserve enclosing return/break/continue targets and the existing depth32 bound, and restore the outer same-named binding after the inner scope. Every owned scope exit still satisfies full-source checks and bytecode affine analysis. I do not mutate parser nodes or erase an unsafe body to bypass checking.

My C body helper can iterate the existing normal/unsafe statement lists without copying AST nodes. My Nano helper factors statement-list traversal from ordinary-block validation, so unsafe traversal can reuse it without appending synthetic AST blocks. Nested scopes increment and restore control depth for lexical name closure and recursion bounds. Owner state is shared deliberately through lexical recursion, while new names end at the scope boundary.

## My acceptance

I first review the complete production delta statically. Then I use a fresh bootstrap and paired C-seed, Stage1 and Stage2 canonical producers with the original source, exact ownership/name metadata, verified VM execution and strict GCC/Clang sanitized native execution. I run the original four-compiler default-driver test and false mandatory-shadow output-preservation control. Additional meaningful cases cover float mutation/copy, shadowed names, zero/entered loop paths, nested scope return/break/continue with owned cleanup, and mixed INT/FLOAT refusal.

Runtime controls establish initialized float local copies, overwrites and joins while owners remain live, all four public VM APIs, repeated success/assertion-failure cleanup and strict native sanitizer/owner-allocation checks. FLOAT parameters/results/resource fields remain checked refusals. I explicitly migrate only PR798's obsolete FLOAT-local refusal control to positive qualification, retaining the original refused result at its old pin and the other three declaration refusals. Tests for uninitialized or inconsistent locals retain admission assertions without executing refused modules.

I retain every first terminal, source/tool inventories and prior output on refusal. This does not admit float owner fields, float call signatures, mixed managed values, callbacks, imports or foreign operations. Full product/fixed-point/platform/release gates remain separate.
