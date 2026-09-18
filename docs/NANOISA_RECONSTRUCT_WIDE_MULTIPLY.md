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

My initial snapshot fixture failed assembly verification because a failure branch retained an extra low word. No module or compiler program was executed. I preserve `/tmp/nanolang-reconstruct-wide-first.log` and `/tmp/nanolang-wide-first-failure`; the corrected fixture stores both words before its assertions. This is a fixture correction, not a runtime failure attribution.

My corrected fixture next reached an explicit Stage1 setup refusal because this isolated checkout lacked `bin/nano_aot_runtime.o`. I preserve `/tmp/nanolang-reconstruct-wide-corrected.log`; I built the local `nvm2c-runtime` prerequisite without modifying my frozen compiler checkout. All four ordinary methods then passed in 23.646 seconds (`/tmp/nanolang-reconstruct-wide-runtime-ready.log`). Compiler tools remain pinned to the successful product; local downstream translator/runtime tools are separately hashed.

## My measured acceptance

At production checkpoint `2621d670`, I passed four focused methods with GCC
(23.646 seconds) and Clang (22.522 seconds). Each positive module ran through
VM, native C translation, reconstructed C with ASan/UBSan, and reconstructed
NanoLang compiled by Cseed, Stage1 and Stage2; the Cseed Nano-C path also had
UBSan enabled. I passed three adjacent small structured-multiplication-loop,
indexed-stack snapshot/loop and compiler-diagnostic methods in 5.016 seconds.
I did not execute the old carry corpus.

I retain logs `/tmp/nanolang-reconstruct-wide-runtime-ready.log`,
`/tmp/nanolang-reconstruct-wide-clang.log` and
`/tmp/nanolang-reconstruct-wide-adjacent.log`. Clang uses the existing GCC13
header-selection wrapper in `/tmp/nanolang-truthiness-clang-bin/cc`.
My producer compiler sources are the successful immutable product pin, not
this generator checkout. My local downstream translator/runtime build is
separately identified below. These are bounded paired results, not a fresh
same-source bootstrap or complete reconstruction result.

I verified these hashes unchanged after all checks:

```text
049a95617d82865327821907a84dece44fd97be4da772183f684713900875631  scripts/nanoisa_reconstruction.py
d930f285ada81cc37f85e4024d115e47e07756ddb81af925f1a509317574935a  /home/jkh/Src/nanolang-nanoisa-only-product/bin/nanoc_c
b299cb96cdbe66c7e1edb6f52539b2af1d9c1634f1d7fb5977fd6abd8f99f128  /home/jkh/Src/nanolang-nanoisa-only-product/bin/nanoc_stage1
e31c7420c16f20b4557d117164e6054c434f5119136b159b878d9d871a2da195  /home/jkh/Src/nanolang-nanoisa-only-product/bin/nanoc_stage2
c2e67d240f1c0028957d982ac2329b8ee51714d91f63c587a8a29113a8717ea4  bin/nvm2hl
15a01ac9b0a4f3465508ba5387b37c5bf0374484f423a0688e839502917e7ff0  bin/nanoisa_hl_facts
ed1a157c63d978dad9e506d00d8f2d0d3b4024dc1099f0a0c738562365839845  bin/nanoisa
daecee302b5ca48c7b1cf75f5e044d441c908e7aad82badfe9a68fbe36da842f  bin/nano_vm
f3921dc5952b9b8d6f2b2cc4652f4154cbf9942ca35cf229c1703b8cb6e7f0c5  bin/nvm2c
411b8b5ccdf1e34ef0187dd8f9eefc78ff39a43fce24c4ca0e4bff71579a5c76  bin/nano_aot_runtime.o
```
