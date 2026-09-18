# My LLVM/Wasm enum scalar contract

I track `task_3762bd4020e94e98be89b83d152f71d8` under 66a6, with typed integer
acceptance also required by a77ca. Native checkpoints 620/623 precede this work.
I admit bounded enum-count metadata and ENUM_VAL in my caller-selected scalar
and literal-string profiles. My general verifier still checks definition bounds.
I keep struct/union layouts, imports, ownership contracts and heap construction
outside these profiles. The ordinal carrier retains tag 9; it is not a pointer.

I match existing VM compatibility rules without redefining enum semantics:
ADD/SUB/MUL/DIV convert enum operands to int before int/float promotion;
I64_ADD/SUB/MUL/DIV_S/REM_S and all six typed I64 comparisons accept int/enum;
MOD/NEG and I64_NEG reject enum. Typed comparisons return bool. Integer wrapping,
zero division/remainder and float division by signed zero keep their contracts.

Enum/int generic equality and ordering compare ordinals. Enum/float equality
remains false and ordering uses tag order. Same-enum generic ordering returns
zero, unlike typed I64 ordering. Truthiness uses nonzero ordinal; CAST_INT returns
the ordinal and CAST_FLOAT returns zero. I preserve original producer tags and
transport enum values through locals, globals, direct calls and checked returns.

CAST_STRING and TAIL_CALL remain explicit pre-existing profile refusals. My
VM/native enum CAST_STRING produces empty text, but this child does not admit
unimplemented generic numeric formatting or managed string operations. I retain
literal-string admission/refusal rules and exact result/parameter tag checks.

I require shared ordinary modules on VM, standalone C, LLVM interpreted and
optimized, sanitized LLVM-native, Wasm and import-free Node where supported.
Controls cover all operations, both orders, actual tags, boundary values,
comparison/cast distinctions, global initialization, calls and invalid-tag
refusals. Invalid profile publication preserves previous output. Full generic
arithmetic and typed-enum parent closure requires the combined evidence, not
only successful LLVM text generation.

My first full same-module gate passed nine of ten enum methods, but the generic
arithmetic matrix could not compile its standalone C product at GCC O2: the tagged
equality fallback triggers a nonnull warning at strcmp. I retain the failure in
`/tmp/nanolang-llvm-enum-focused.log` and `full.log` under the same prefix.
I record native prerequisite `task_39453e3f2c76454dab9afd3e346c2483`; I do not
weaken flags or count that failed gate as complete. PR627 repairs this prerequisite
with VM-matched pointer/null equality before strcmp. Managed string/cast coverage stays
under my roadmap's managed-string lifetime/allocator/linkage obligation.

## My arithmetic boundary

My enum integer compatibility applies to ADD/SUB/MUL/DIV and the eleven typed
binary I64 operations. It does not make enum a universal numeric alias. Generic
MOD/NEG and typed I64_NEG still refuse enum. I preserve binary64 rounding when an
enum is combined with float, including 2^53 plus ordinal 65535 rounding to
9007199254806528. A zero enum divisor returns integer zero for integer/enum
numerators and positive float zero for float numerators, including NaN and
infinity. Nonzero float operations retain IEEE NaN behavior.

I retain the preceding scalar arithmetic evidence from
[total integer arithmetic](evidence/native-total-integer-arithmetic.md),
[native numeric promotion](evidence/native-generic-numeric-arithmetic.md),
[native tagged arithmetic](evidence/native-tagged-numeric-arithmetic.md),
[NATIVE_NUMERIC_UNION_SHAPES.md](NATIVE_NUMERIC_UNION_SHAPES.md), and
[NANOISA_LLVM_GENERIC_NUMERIC.md](NANOISA_LLVM_GENERIC_NUMERIC.md).
The enum companion supplies the remaining enum arithmetic obligation for 66a6
and the LLVM/Wasm typed-enum obligation for a77ca. Full heap, host, managed string
and target coverage remain separate. These are tested compatibility boundaries,
not a proof of backend equivalence.

My expanded gate also exposed an older U8 test that required native translation
refusal even though current boxed return transport emits an exact runtime tag
check. I record `task_1dfec6598e464afbbcdeb1885a3694b6`, retain the failure in
`/tmp/nanolang-llvm-enum-repaired-full.log`, and require the generated ordinary
program to stop at the native invariant without sanitizer errors. This changes
only the test expectation; it does not weaken the return guard.

## My measured acceptance

On main `79e156ac` with source/test checkpoint `4539c521`,
`make test-nvm2llvm test-nvm2wasm test-verifier-profiles` passes 97 methods:
16 LLVM scalar/F64, 3 managed-core, 11 enum, 11 scalar-global, 9 literal-string,
7 generic-numeric, 39 combined Wasm/truthiness/U8/implicit/comparison, and
1 profile method. I retain `/tmp/nanolang-llvm-enum-final-full.log`.

The final test addition checks two distinct enum declaration IDs with equal and
unequal ordinals, typed and generic comparison differences, arithmetic result
tags, global/local transport and a checked enum-return helper. This same-module
VM/C/LLVM/Wasm/Node method passes separately. It and the corrected U8 return
check also pass with strict Clang native C plus ASan/UBSan/LSan. I retain
`/tmp/nanolang-llvm-enum-declarations.log` and
`/tmp/nanolang-llvm-enum-clang-final.log`. Earlier refusal tests check wrong-tag
results independently. My final enum suite contains twelve methods; the full
97-method run preceded only that last test-only addition.

These results complete the bounded scalar arithmetic acceptance shared by 66a6
and a77ca together with their preceding native/LLVM checkpoints. They do not
complete managed formatting, heap arithmetic, TAIL_CALL, host imports, or full
LLVM/Wasm language coverage, and make no new full compiler-product claim.
