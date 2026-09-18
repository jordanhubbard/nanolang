# My LLVM/Wasm enum scalar contract

I track `task_3762bd4020e94e98be89b83d152f71d8` under66a6, with typed integer
acceptance also required by a77ca. Native checkpoints620/623 precede this work.
I admit bounded enum-count metadata and ENUM_VAL in my caller-selected scalar
and literal-string profiles. My general verifier still checks definition bounds.
I keep struct/union layouts, imports, ownership contracts and heap construction
outside these profiles. The ordinal carrier retains tag9; it is not a pointer.

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
