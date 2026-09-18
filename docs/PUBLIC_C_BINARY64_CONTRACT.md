# My public C scalar binary64 contract

I start local child `task_54c918adf5c04bb78e420c5881c95e82` on canonical main
`c67eef7c5bdd5664e6b63a614a6d04513af10e8d`. Parent070db retains its failed
worker/FFI evidence. I do not reuse that worker's claimed acceptance. PR740's
source reduce child8618 is reconciled by canonical ancestry; d099 still retains
canonical native FUNCREF, and scalar policy5009 stays open.

## My observed source boundary

The public C-seed --target c route calls src/c_backend.c after source typechecking.
It is distinct from native legacy --keep-c and the selfhost C emitter. Its header
promises self-contained C99/C11. Static inspection identifies these prerequisites:

- infer_expr_type defaults calls to INT, using spelling guesses for strings.
  Prefix arithmetic follows only the first operand. The emitter has no exact
  builtin bit-transport handling and emits float constants with six-digit %g.
- Binary operations print raw C operators. Function argument evaluation order
  does not establish source left-before-right arithmetic evaluation.
- emit_program handles type/function declarations but skips global let bindings.
  Dynamic scalar globals cannot simply become C static helper initializers.
- c_backend_emit opens the final path with w before checking the emitter result.
  A checked refusal can therefore destroy previous output.
- main uses the ordinary int64_t return mapping, unlike the hosted C int entry
  signature. Shared binary64 helper guards currently use C11 _Static_assert.
- Existing expression-block emission uses GNU statement expressions. This is a
  separate wider portability boundary, task_6ade6d62ef644390bb9645b15077c8df.
  I neither use that mechanism for new scalar sequencing nor claim that all
  pre-existing target features/options are already portable.

I inspect these sources only. I do not regenerate or execute retained failed
compiler/callback artifacts.

## My staged implementation contract for review

1. **Publication and scalar planning.** I stage path output in an owned temporary
   file and rename only after successful emission and I/O close checks. I retain
   the first useful diagnostic, clean owned temporary state on refusal, and keep
   later valid compilations independent. The FILE API stages semantic emission
   before copying to its caller-owned stream; it cannot promise rollback after
   an external stream write failure. I collect declared function results, scoped
   parameters/locals and scalar globals. Scalar arithmetic/bit operations require
   exact resolved types; unknown calls do not inherit an INT guess. A bound name
   does not become an intrinsic by spelling. I retain existing source checker
   policy for bit intrinsic name reservation and do not change that policy here.

2. **Exact scalar values and ordered operations.** I emit every float literal from
   its integer binary64 representation, with memcpy construction and representable
   signed-pattern conversion. I lower float_from_bits(INT)->FLOAT and
   float_to_bits(FLOAT)->INT without FP arithmetic or numeric casts. I preserve
   every payload/sign bit, including signaling NaNs. I reuse the reviewed shared
   scalar add/sub/mul/div helpers, canonical positive quiet NaN result bits
   0x7ff8000000000000, and signed-zero divisor precedence returning positive zero.
   Unary negation, comparisons, transported inputs and external math are not
   canonicalized. Existing source numeric promotion rules remain authoritative;
   this child introduces no enum/heap/array conversion policy.

   Each admitted binary scalar arithmetic expression receives distinct automatic
   operand temporaries in its enclosing function or initializer function. A C
   comma expression sequences left assignment, right assignment, then helper call.
   Nested expressions use distinct slots; loops execute assignments each iteration;
   recursive calls receive their own automatic storage. I do not hoist operand
   evaluation out of branches or loops. The temporary/helper namespace is selected
   against actual user declarations and local bindings, not a silently reserved
   ordinary name. I preserve known user f64_* names and temp-like identifiers.
   I neither use GNU statement expressions nor assume C argument order.

   Shared helper storage assertions gain an equivalent C99-compatible form while
   preserving the same binary64/fast-math/evaluation guards and operation bodies.
   I update generated helper providers mechanically and test their identity.
   This assertion compatibility change does not weaken any existing target guard.

3. **Ordered globals and entry.** I emit scalar storage and prototypes before a
   private initialization function. I evaluate global initializers once in source
   declaration order before user main, preserving immutable/mutable language
   bindings, chained dependencies and stateful calls. Global operand temporaries
   belong to that initialization function. I use the existing hosted C int main
   ABI without changing language int storage elsewhere. Unsupported dynamic global
   entry/library combinations receive a checked diagnostic rather than silently
   missing initialization. I do not claim thread-safe library initialization or
   implement unrelated heap/global import expansion in this child.

4. **Qualification.** I send production checkpoints for independent review before
   fresh ordinary fixtures. I freeze production, harness and tools for each gate.
   I compile actual public --target c output with GCC and Clang under strict
   C99 and C11 (pedantic errors), O0/O2 and UBSan, keeping host/compiler identities.
   Integer observers cover all four scalar operations, NaNs, signed zero, infinities,
   subnormal/rounding boundaries and non-contraction. Once-only stateful operands,
   nested arithmetic, both conditional arms, loop reevaluation, local shadowing,
   function-return typing, user helper-like names and ordered globals receive
   separate controls. Wrong-type/unsupported expression controls preserve prior
   output; a later valid compile in the same process must recover. I compare fresh
   ordinary sources with the already-qualified interpreter/legacy/VM/native routes
   where their existing profile supports the same source; no old failure replay.

   I explicitly refuse a newly affected scalar expression form if I cannot lower
   it in C99/C11 with resolved types and correct evaluation, then record that
   residual requirement. Such a refusal cannot count as full scalar-policy success.
   I do not silently replace the public backend with the legacy GNU emitter.

## My completion boundary

I close local child54c918 only after its reviewed bounded implementation and
qualified evidence merge. Parent070db requires the actual supported public scalar
route, including its bit-observable arithmetic and ordered globals, before closure.
Unmet scalar cases remain explicit prerequisites. Wider C expression/options
portability6ade, canonical FUNCREF/d099, array arithmetic3717 and full release
remain independent. Parent5009 stays open until all of its remaining source and
callback criteria are met.

My post-build publication audit keeps the first semantic failure and adds explicit
first diagnostics to remaining allocator/stream/rename failure returns. I preserve
owned temporary cleanup and the caller-stream partial-I/O limit. My final fresh
gate also covers a global initializer calling main while initialization is in
progress, alongside ordinary main reentry after initialization.

My first seven-method gate retains six passing methods and one C API harness link
failure: the isolated fragment omitted passive_binding_order. That fragment has
no passive AST cases; I add an explicit unreachable fixture stub, and leave real
passive qualification with the existing backend suite. No failed API binary ran.

Independent static review also finds that directly changing the user main return
to C int would narrow same-module language calls to main. Before further execution
I keep user main as a private int64 function and expose a separate hosted int
wrapper. I map declarations, direct calls and unshadowed function references
consistently under the selected private prefix; local main bindings stay local.
I add a fresh global-initializer/main-reentry high-bit result observer. My previous
zero-result reentry controls do not qualify this wider result boundary.
