# My scalar global LLVM/Wasm checkpoint

I tested source `391c4441` after integrating main `c9cabb70`. MAC
`task_2baf4ba2b74b4d5aa94fb31cc9e81356`.

I deliberately extend closed-scalar admission with LOAD_GLOBAL, STORE_GLOBAL and
a zero-argument selected initializer. My LLVM table is sized from all verified
literal references and bounded by the existing 4096-slot verifier limit. Every
slot starts as void; functions share exact tagged last-write values. The first
named initializer runs before entry and its result is discarded. Global storage
is not reset by repeated entry calls.

My eleven global methods pass on Linux ARM64. They exercise initial void,
slot 4095, an uncalled function's storage requirement, int/U8/float/bool/void
replacement, NaN transport, helper calls, branch-selected writes, all five scalar
initializer result forms, initializer failure, first-name selection and refusal
before output replacement. Ordinary modules run on VM, LLVM, optimized LLVM,
Clang-native LLVM with ASan/UBSan and Wasm.

I separately invoke `vm_execute` twice on one VM instance and once on a fresh
instance, then compare repeated exports from one Wasm instance and a fresh
instance. Both produce `1,2,1` with or without an initializer. When initializer
and entry name the same function, both produce `2,4,2`. LLVM and sanitized native
observer entry calls match the two-call result. These are same-instance lifetime
checks, not repeated fresh process executions.

My text assembler requires unique function symbols. To check the VM's first
initializer-name selection, a test-only public module API fixture assigns the
same display name to a second function and serializes a normally verified
module. The second function has an argument and a failing assertion; it is not
implicitly called. VM/LLVM/native/Wasm all run the first initializer only. No
product assembler restriction changes.

The integrated target gates passed 16 existing LLVM methods (23.059 seconds),
11 global methods (6.975 seconds), seven generic numeric methods
(123.433 seconds), 39 Wasm/adjacent methods (77.730 seconds), and the 12-case
shared profile method (0.308 seconds). Independent production review
found no scoped blocker. No new wire metadata, host source dependency or compiler
`.nano` source is introduced.

I retain import, heap, nominal, ownership/passive and capture restrictions. I do
not claim concurrent or linked-module global semantics, full target coverage,
or release acceptance. Those parent obligations remain open.
