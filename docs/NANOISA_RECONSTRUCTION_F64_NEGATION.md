# My typed binary64 negation reconstruction contract

I record `task_945d1fdcaaee499daddf1e0187e70a2b` before implementation under
full reconstruction parent `task_4bd034f6029b7458201db74e2c3aeb32`.
My starting main is `a0a8fbcf`, containing PR726 typed comparisons and the
previous exact binary64 facts, text and transport prerequisites.

My bounded implementation is qualified in [my evidence](evidence/reconstruction-f64-negation.md).

## My operation and source boundary

I propose only F64_NEG: one exact FLOAT operand and one FLOAT result. I retain
separate refusal for generic NEG with FLOAT, typed I64 negation with FLOAT,
mixed/unknown tags and every other unimplemented float operation.

My current VM uses `val_float(-a.as.f64)` after an exact tag check. My native
translator emits unary minus on FLOAT storage; LLVM emits `fneg`. I reconstruct
ordinary unary C minus and unary Nano `(- operand)` with explicit FLOAT
annotations. I do not replace negation with subtraction from zero, numeric
casts, formatting or generic arithmetic. I inspect each canonical producer's
output for the dedicated F64_NEG opcode.

My required observable result reverses the binary64 sign bit and retains all
other bits, including NaN payload and signaling/quiet encoding. This includes
both signed zeros, finite/subnormal values and infinities. I construct inputs
through the exact integer-decoded facts path and qualified bit intrinsics.
My test oracle uses signed integer bit observations, never float equality,
Python float conversion or formatted text. Double negation recovers the original
pattern. The original operand snapshot remains unchanged after either result.

I qualify this exact-bit requirement against the existing VM and all admitted
source producer routes before completing admission. If an ordinary qualified
route changes a NaN encoding, I retain the first failure and record a separate
prerequisite rather than weaken the oracle or claim universal host behavior.
I make no floating-exception-environment promise beyond ordinary default
execution and do not use fast-math flags.

## My unchanged invariants

I preserve immutable per-instruction snapshots, once evaluation of effectful
operand calls and evaluation of discarded results. Locals and direct helper
arguments/results remain exactly typed. Existing acyclic calls, explicit
returns, definite initialization, empty-stack joins and pure-loop restrictions
remain intact. My entry stays zero-argument INT; no heap or tag widening occurs.

## My qualification order

1. I freeze both production and the complete test harness before starting gates.
   If a correction is needed, I retain its prior outcome and run the affected
   final harness gates after the corrected checkpoint; I do not mutate live tests.
2. I use fresh small independent modules with signed zero, minimum/maximum
   subnormal, normal and finite boundaries, infinities, signed quiet/signaling
   NaNs and varied payloads. I check result bits, original operand bits and
   double negation through exact integer observations.
3. I exercise helper transport, later stores after a snapshot, discarded-call
   evaluation, both branch arms and a bounded pure loop. Test-only counters
   check once evaluation without admitting global opcodes.
4. I compare original VM/native with reconstructed standalone C under GCC and
   Clang O2 strict warnings and ASan/UBSan. Qualified PR720 C-seed, Stage1 and
   Stage2 compile reconstructed Nano; NanoVirt/Stage1/Stage2 also emit canonical
   modules for VM/native execution and typed opcode inspection.
5. I verify producer/import hashes before and after gates and record current
   generator/tool identities separately. I preserve previous outputs on remaining
   unsupported-operation refusals and keep exact other-tag analyzer controls.

I retain qualified PR720 source tools unchanged and do not claim a fresh
bootstrap or hermetic relocation. I do not replay historical PR679 artifacts
or modify frozen product acceptance. Binary float arithmetic, numeric casts,
float truthiness, generic FLOAT operations, heap reconstruction and full parent
closure remain separate work. I push this contract for independent review
before production admission.
