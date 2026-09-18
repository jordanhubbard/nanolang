# My exact wide-product reconstruction

I record `task_684295d1eb064fecb03685bf65bb0697`, a child of reconstruction
parent `task_4bd034f6029b7458201db74e2c3aeb32`, before implementation.

I admit only `I64_MUL_WIDE_U` and `I64_MUL_WIDE_S` with two exact integer
operands. I preserve the VM's low-word then high-word stack order. Ordinary
operand evaluation and both result words receive immutable snapshots; calls
are not repeated to obtain the second result. Pure loop expressions retain
my existing no-call/no-store restriction. I add no tuple/heap result ABI.

I split each unsigned operand into four base-65,536 limbs. Seven convolution
coefficients plus their carried final digit form the 128-bit product. A limb
product is below 2^32; a coefficient plus carry is below 2^34; a carry is
below 2^18. These bounds keep every NanoLang multiplication and coefficient
addition representable as signed int. I normalize the signed low remainder
and use logical shifts for higher limbs. I pack words with existing total
arithmetic helpers. C uses unsigned intermediates and representable signed
bit reconstruction, without requiring a 128-bit C extension.

The low word is the existing total 64-bit product. Signed high equals
unsigned high minus `b` when `a` is negative, minus `a` when `b` is negative,
all modulo 2^64. I use total subtraction for those corrections. Finite helper
cost is an implementation bound, not a performance-parity claim.

I require fresh small programs checking both words against independent
integer arithmetic, signed endpoints, cross-word carry, stack order,
call/local snapshots, loops, exact-tag/arity refusals, unchanged prior output,
and byte-identical dump/reassembly. I compare VM, reconstructed C and
NanoLang compiled by three known successful producers. I retain the first
failure if one occurs. I do not regenerate carry679's historical endpoint
corpus or execute its failed artifacts or historical compiler selections.

My selected immutable compiler checkout is
`/home/jkh/Src/nanolang-nanoisa-only-product` at `f38b6409`, with recorded
production `ece241ba` and completed bootstrap/product acceptance in
`docs/evidence/product-gates-ece241ba-complete.json`. I verified Cseed,
Stage1 and Stage2 hashes against that completed evidence before testing.
Full reconstruction, source-pair and normative equivalence remain open.
