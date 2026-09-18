# My typed binary64 comparison reconstruction contract

I record `task_4b603437748343d48a5f92bf577bc5be` before implementation under
full reconstruction parent `task_4bd034f6029b7458201db74e2c3aeb32`.
My starting main is `55208c47`, containing PR723 exact constant/bit transport.
This document proposes admission; it does not claim completed acceptance.

## My exact boundary

I admit only F64_EQ, F64_NE, F64_LT, F64_LE, F64_GT and F64_GE. Each consumes
two exact FLOAT values and produces BOOL. I reject INT, BOOL, mixed tags and
unknown tags at this analyzer boundary. I do not change generic comparison
handlers or typed I64 rules.

My existing VM dispatch in `src/nanovm/vm.c` checks two FLOAT tags and uses the
six C relational operators. For either quiet or signaling NaN operand, my
observable BOOL result is false for EQ/LT/LE/GT/GE and true for NE. Signed zeros
compare equal; infinities and finite/subnormal values retain binary64 ordering.
Generic comparisons have a different unordered ordering contract and remain
excluded for FLOAT. I do not implement typed comparisons via generic ordering,
bit-pattern ordering or subtraction.

I render ordinary C relational operators and Nano prefix operators with exact
FLOAT annotations. The C-seed `src/nanovirt/codegen.c` float operator branch and
selfhost `nisa_float_mnemonic` already select these dedicated F64 opcodes. I
verify that producer selection in acceptance rather than assuming generic
operators are equivalent. I use no fast-math flags.

My operand constants retain the PR723 sixteen-hex-digit facts path and exact
bit intrinsics. I preserve every operand snapshot before a later store and
once evaluation of operand calls, even when a result is predictable from the
constant inputs. Comparisons produce BOOL and never replace their source
FLOAT slots. I observe original operand bits after comparison through exact
integer observers. I make no floating-exception flag or trap-mode guarantee;
my execution contract uses the existing ordinary default floating environment.

I retain acyclic direct calls, explicit returns, exact local tags, definite
initialization, empty-stack structured joins and conservative pure loop
conditions. I do not admit effectful condition calls or new stack joins.
My entry stays zero-argument INT. Arithmetic, negation, numeric casts, float
truthiness, generic FLOAT comparisons, heap values and mixed numeric promotion
remain separate dependencies.

## My acceptance order

1. I add exact analyzer and emitter cases with other-tag and unknown-op refusal
   controls. Refused modules retain existing output files and do not execute.
2. I use fresh small modules covering all six operations, both operand orders,
   signed zeros, finite/subnormal boundaries, infinities, and signed quiet and
   signaling NaNs with differing payloads. Expected comparison results are
   explicit BOOL/integer observations, not reconstructed float equality used
   as a bit-preservation oracle. NaN constants remain integer-decoded.
3. I exercise helper arguments/results, snapshots before later local stores,
   both structured branch arms, and bounded pure loop conditions. Test-only
   call counters establish once evaluation without admitting global opcodes.
4. I compare original VM/native with standalone reconstructed C under GCC and
   Clang with ASan/UBSan. Reconstructed Nano runs through qualified PR720 C-seed,
   Stage1 and Stage2 legacy compilation and both canonical producer
   implementations, followed by VM/native execution. I inspect typed opcode
   selection in generated modules.
5. I preserve and verify PR720 tool/import hashes before and after tests using
   the prior manifest. I identify reused producer tools separately from the
   reconstruction source under test; I do not claim a new bootstrap or hermetic
   relocation. I retain the first failure and qualify any correction separately.

I preserve historical PR679 artifacts and frozen product trees. This child
cannot close full reconstruction or release acceptance. I push this contract
for independent review before changing production admission.
