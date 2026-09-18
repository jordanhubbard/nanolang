# My generic numeric LLVM/Wasm checkpoint

I tested source `642915b6` after integrating main `d0f72962` (including paired
source borrows609 and native known-numeric arithmetic610). MAC
`task_fd74e0169a1c4be0ae328ead691bde23`, under parent
`task_66a6dd8ca51d415f9efb0f2904f85b49`.

I add only ADD/SUB/MUL/DIV/MOD/NEG to the shared closed-scalar profile with
matched LLVM lowering. Both numeric tags are checked before any float promotion.
Integer operations retain wrapping and total division; mixed operands promote to
binary64. Modulo remains integer-only; negation preserves integer/float tags.
LLVM operations carry no signed-overflow or fast-math assumptions. Wasm consumes
the same verified LLVM lowering.

My seven new methods execute 41 ordinary modules across VM, LLVM, optimized LLVM,
Clang-native LLVM with ASan/UBSan, and Wasm. They check typed results, both
promotion orders, integer extremes, binary64 precision, NaN/infinity, calls,
dynamic local joins and eager evaluation. Thirty-six operand cases intentionally
fail at runtime; a separate right-operand assertion checks eager evaluation.
These programs pass profile admission and publish successfully before runtime
failure. That differs from heap/profile refusal, which preserves prior output.

Five additional cases compare exact result bits from VM host invocation with an
observation-only wrapper around the unchanged emitted helper on LLVM, sanitized
native and Wasm. They distinguish NEG of positive/negative zero, negative-zero
multiplication, and positive-zero division results for NaN or infinity divided by
negative zero. The wrapper is a test harness, not new product functionality.
Its first Wasm export named `main` hit the toolchain's special entry adapter;
using `nano_bits_check` corrected the harness without changing production code.

On this integrated source I passed the new seven methods (233.669 seconds),
the existing 16 LLVM methods, 39 Wasm/adjacent methods, and the 12-case
verifier-profile gate. `make test-nvm2wasm` now requires the new numeric gate.
Independent source review found no scoped blocker. No source compiler `.nano`
files changed in this slice, and I make no fresh compiler-bootstrap claim.

I retain heap, enum, import, layout and ownership refusal in this scalar profile.
Native C known-shape promotion landed separately; its broader tagged-value
promotion remains separate work. Full arithmetic and target-coverage parents
remain open.
