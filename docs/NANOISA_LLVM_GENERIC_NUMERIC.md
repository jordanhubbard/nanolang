# My generic numeric LLVM/Wasm contract

I extend my caller-selected closed-scalar profile only with matched lowering of
`ADD`, `SUB`, `MUL`, `DIV`, `MOD` and `NEG`. My general verifier is unchanged.
MAC `task_fd74e0169a1c4be0ae328ead691bde23`, under arithmetic parity parent
`task_66a6dd8ca51d415f9efb0f2904f85b49`.

I audited `src/nanovm/vm.c` at main `7ec1ff5a`. For `ADD/SUB/MUL/DIV`, two
integers produce an integer; either float promotes both operands to binary64.
Integer add/subtract/multiply/negate wrap modulo 2^64. Integer division and
remainder by zero return zero; minimum integer divided by -1 returns minimum,
and its remainder is zero. Float division by either signed zero returns positive
zero, including a NaN numerator. Other float arithmetic follows the existing
IEEE operations without fast-math. `MOD` accepts only integers. `NEG` accepts
an integer or float, preserving float signed-zero behavior.

I reject bool, U8 and void arithmetic at runtime, as my VM does. I admit no new
heap, enum, import, layout or ownership instructions. Existing LLVM/Wasm module
and signature restrictions remain. This is tagged numeric lowering, not the VM's
string concatenation, array arithmetic or enum-coercion support.

My tests use the same ordinary assembly modules on VM, LLVM interpretation,
optimized LLVM, native compiled LLVM with sanitizers, and Wasm. I check result
values and tags, promotion in both operand orders, precision boundaries, NaN and
signed zero, integer extremes, calls, joins and runtime type errors. I retain
heap refusal and old-output preservation. I coordinate C native known-shape
arithmetic with its separate child; I do not close the full arithmetic parent.
