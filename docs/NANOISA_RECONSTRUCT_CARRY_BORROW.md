# My carry and borrow reconstruction contract

I record task_cbdce24cc6b747f6b39ceb4c87f20676 before implementation under parent task_4bd034f6029b7458201db74e2c3aeb32. I admit I64_ADD_CARRY and I64_SUB_BORROW with exactly three INT operands. I normalize the input carry/borrow to its low bit, including negative carriers, and push the wrapped low result followed by an INT high result of zero or one, matching my VM.

I represent the two results as exact scalar expressions with distinct per-instruction low/high temporary names. Existing operand snapshots retain evaluation order and once-only direct calls. Pure loop-condition expressions remain side-effect-free. I do not introduce aggregate storage, tuple return types or a multiresult function ABI.

My C helpers use uint64_t arithmetic and representable signed bit conversion. NanoLang helpers use my tested total add/subtract and unsigned comparison helpers, with explicit zero/one high results. I retain previous output on exact-tag/arity/unsupported-profile refusal. I require both result orderings/consumption, endpoint and noncanonical carry inputs, calls/snapshots/loops, canonical byte roundtrip and paired VM/reconstructed C/three pinned NanoLang compiler execution with GCC/Clang sanitizers. Source and tool hashes remain separate; full reconstruction and wide multiplication remain open.
