# My typed binary64 arithmetic reconstruction contract

I record `task_509569917fdd4ce390a67e5d5b1e55f1` before implementation under
full reconstruction parent `task_4bd034f6029b7458201db74e2c3aeb32`.
I start from main `90bd4787`, after PR763's original scalar/callback acceptance
matrix. I implement locally without dispatch and request independent contract
review before production edits.

My bounded implementation and actual producer pins are recorded in
[my evidence](evidence/reconstruction-f64-arithmetic.md).

## My exact operation boundary

I add only F64_ADD, F64_SUB, F64_MUL and F64_DIV to my existing scalar region
analyzer. Each consumes right then left from the stack, requires two exact
FLOAT values and produces one FLOAT. Mixed, unknown and non-FLOAT inputs remain
refused. I do not widen generic ADD/SUB/MUL/DIV, casts, truthiness, heap shapes,
imports, globals, ownership or callable-value reconstruction.

I use my [shared arithmetic policy](NANOISA_BINARY64_ARITHMETIC_POLICY.md):
ordinary binary64 rounding, arithmetic NaNs normalized to `0x7ff8000000000000`,
and F64_DIV by either signed zero returning positive zero before inspecting
the numerator's arithmetic result. This includes NaN and infinity numerators.
I preserve operand payload bits and my existing exact transport/negation rules.
I require the default nearest-even environment, gradual underflow and no
fast-math, excess-precision evaluation or contraction across instructions.
This bounded work does not establish alternate rounding-mode semantics.

## My emitted source and evaluation order

I retain the analyzer's immutable per-instruction snapshots. Calls and operands
are evaluated once in original instruction order, including discarded results;
later local writes cannot change earlier operands. Every binary result is
materialized at its instruction boundary. My existing pure expression/loop
rules remain unchanged and cannot elide effectful calls.

My standalone C output embeds the current exact `src/binary64_arithmetic.h`
text when these operations are used, then calls its `nano_rt_f64_add`, `sub`,
`mul` and `div` helpers. I do not reconstruct an approximate local replacement
or depend on a repository include at C compile time. I test embedded source
identity and retain its target/fast-math guards and rounding boundaries.

My Nano output uses ordinary typed FLOAT prefix operators on those snapshots.
Fresh post-policy producers must emit the corresponding typed F64 opcodes;
source spelling alone is not evidence. Exact bit intrinsics observe results
without decimal formatting or conversion. My external mandatory-shadow harness
is validation support, not recovery of the original module's source shadows.

I preserve exact scalar helper signatures, acyclic direct calls, explicit
returns, definite initialization, empty-stack joins, bounded pure loops and
zero-argument INT entry. Existing output publication stays atomic on refusal.

## My fresh qualification

1. I freeze source, tools and test harness before each gate. I build current
   C-seed/runtime tools and perform a fresh Stage1/Stage2 bootstrap from the
   implementation's post-policy source pin. I record that actual bootstrap pin,
   imported source/library hashes and every tool hash before and after tests.
   A later documentation-only integration does not become a new bootstrap claim.
2. I construct fresh small modules using integer bit patterns for signed zero,
   finite endpoints, minimum subnormal, normal/subnormal boundaries, infinities,
   signed quiet/signaling NaNs and distinct payloads. I observe exact results
   and unchanged inputs for all four operations. I cover overflow, underflow,
   cancellation, nearest-even ties, signed-zero divisor precedence and separate
   rounding of a multiply followed by addition.
3. I exercise both branch arms, a bounded loop, helper arguments/results,
   later local overwrites, discarded calls and both operand evaluation orders.
   Test-only instrumented source counters check once evaluation without adding
   global opcode admission. I retain exact-tag analyzer refusals and previous
   output bytes when unsupported operations are rejected.
4. I compare original verified VM and native execution with reconstructed
   standalone C under GCC/Clang O2 strict warnings and ASan/UBSan. Fresh C-seed,
   Stage1 and Stage2 compile the reconstructed Nano through their supported C
   routes; fresh NanoVirt, Stage1 and Stage2 publish canonical modules for VM
   and native execution and typed opcode inspection. Fixed expected integer
   bits and original-policy results form the oracle, not formatted float text.
5. I run relevant existing reconstruction analyzer/emitter controls with the
   same fresh tool selection, and the shared helper generation identity check.
   I preserve each first failure, correct only an established cause, then run
   affected final gates from a newly frozen checkpoint.

I leave old f38 tools/libraries and historical PR679 failure artifacts immutable
and unexecuted. I do not inherit an older test invocation's producer selection;
my harness records and explicitly selects the fresh tools. Full reconstruction
and release acceptance remain separate from this four-opcode child. I send
this contract for review before production, then submit the implementation and
its bounded evidence through a pushed PR.
