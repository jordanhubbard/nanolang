# My typed unsigned division reconstruction contract

I record task_d0b4b473329745cd848694ec835965ad before implementation, under full reconstruction parent task_4bd034f6029b7458201db74e2c3aeb32. I admit only exact INT operands/results for I64_DIV_U and I64_REM_U. I interpret their signed carriers as unsigned 64-bit patterns, matching my VM dispatcher: division and remainder by zero return zero. All other quotients and remainders return their exact bits in signed carriers.

My C helpers use uint64_t arithmetic and my existing representable signed-bit conversion. My NanoLang helpers use at most 64 binary long-division steps, exact unsigned comparison, logical shift and total add/subtract helpers. A saved high-bit carry distinguishes an overflowing doubled remainder before subtraction. I never take abs(MIN), rely on signed overflow, or assume unsigned source syntax. This finite helper cost is not a performance-parity claim.

I require endpoint/high-bit/zero pairs, calls/loops/evaluation snapshots, strict tag/arity and previous-output refusals, byte roundtrip, and the same module through VM, reconstructed C and three pinned NanoLang compiler tools. I retain generator source hashes separately from reused compiler tools, run GCC/Clang sanitizers, and preserve initial failures. Full reconstruction, generic numeric operations and heap/metadata expansion remain separate.
