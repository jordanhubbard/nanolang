# I qualify owned-profile binary64 operand-stack scalars

I record `task_cb823e6c37e6465e8d6ad10f3b726364` before production under the
mixed ordinary/owned proposal `task_4be28fef163f42069064357639b3b5cc`.

My inner affine analyzer supports PUSH_F64, float operand-stack copies, F64_ADD/SUB/MUL/DIV,
F64_NEG, six F64 comparisons, and generic EQ/NE/LT/LE/GT/GE with two matching scalar
tags. My native owned emitter lacks F64 cases and generic comparisons currently
read integer storage. My initial claim that the complete public verifier already admitted these instructions
was incorrect: its runtime whitelist rejects them. I retract that claim and
require the companion below before execution. I keep existing
source, parameter/result, resource-field, reference and managed-value admission.

I add a distinct binary64 scalar carrier member and explicit runtime tag to the
private generated value. Every literal/result producer establishes its actual tag;
LOAD/STORE/DUP/SWAP, owner field transport, reference access and calls preserve it.
A borrowed scalar call result establishes the callee's already-checked result tag.
I do not guess a float tag from stack depth or integer storage. Generic comparison
chooses scalar semantics from these tags after existing exact verifier admission.

I transport literal bits with memcpy from the decoded uint64 representation,
without decimal formatting or signed integer casts. Copies preserve those bits.
F64 binary arithmetic uses my existing shared nano_rt_f64_* provider: rounded
binary64 results, canonical arithmetic NaNs and either signed-zero divisor yielding
positive zero. F64_NEG follows the VM unary operation. F64 predicates use direct
IEEE comparison: NaN is unequal and all ordered predicates are false.

My generic float equality uses C floating equality, as vm/value.c::val_equal does.
Generic ordering uses the VM three-way comparison: a<b gives -1, a>b gives 1,
otherwise 0. Consequently NaN generic LE/GE are true, whereas F64_LE/F64_GE are
false. I preserve that existing distinction, including signed-zero equality.
Mixed INT/FLOAT generic operands remain rejected by affine_bytecode's matching-tag
rule; I do not silently broaden it to the VM's ordinary mixed-tag behavior.

My first production checkpoint precedes execution and receives independent review.
After review I require ordinary newly generated valid modules containing actual
owner operations plus float scalar work, VM/native matching results, every generic
and F64 comparison across signed zeros, finite values, infinities and NaNs, and
arithmetic/transport bit observations through the generated private carrier where
public float returns remain refused. I include scalar local copies, float stack copies, branches, loops,
existing owner field observation, helper scalar results and existing
integer/bool/u8 controls. No unsupported module is executed.

I run strict GCC and Clang compilation, applicable sanitizer/leak controls and the
existing owned graph/result/reference regressions. I preserve first terminal
outcomes and source/tool identities. No Samples/Bundle/source-family completion or
new managed runtime/LLVM/Wasm admission follows from this prerequisite.

I corrected my initial local-copy assumption after the new fixture was refused at
fe190992 before execution: the outer owned verifier excludes FLOAT locals and
resource fields, even though inner affine analysis supports their scalar tags.
I preserve that outer boundary. My float loops carry values on the operand stack
with integer counters; no FLOAT local, field or function signature is admitted.


## My explicit public admission companion

I record `task_77952f5657c94f0181354a8193d0d5ec` after the corrected e7cb4fce
fixture was refused by `owned_runtime_opcode`, before execution. I retain that
terminal log, all source hashes and the preceding float-local refusal separately.
I propose adding exactly PUSH_F64, F64_ADD/SUB/MUL/DIV/NEG and the six F64
comparisons to this outer whitelist. Generic comparisons were already listed;
the inner affine analyzer still requires two matching exact scalar tags.
I do not add FLOAT to local, parameter, result or resource-field whitelists.

I inspected the complete public activation chain in `src/nanovm/vm.c`:

* `vm_module_ownership_required` distinguishes advisory metadata from required
  ownership; `vm_module_ownership_supported` requires the complete public owned
  verifier for required standalone execution. Linked owned modules stay refused.
* `vm_ownership_admit` checks constant readiness and obtains only a synchronous
  invocation proof after that complete verifier. Callback/trace/inherited-reference
  paths use the same complete module verifier rather than publishing fast proof.
* Public `vm_core_execute` calls scoped core without a proof. Scoped core always
  checks runtime readiness, uses full admission without a matching invocation
  proof, then checks frame count/module/generation/activation invariants.
* `vm_call_function` creates a fresh proof and uses the scoped admission/cleanup
  wrapper. `vm_execute` reaches it. `vm_invoke` and `vm_invoke_callable` each start
  with fresh admission, retain exact public entry restrictions and unwind roots
  and reference activations on failure. I change none of those paths.
* Shared PUSH_F64 pushes the decoded value through existing stack machinery.
  F64 binary/unary/predicate handlers require exact FLOAT tags; arithmetic uses
  the shared provider, predicates the existing VM comparison policy. These new
  admitted scalar instructions allocate no heap objects and cannot mint owner or
  reference authority. Structural/stack checks and affine tag/edge checks still
  precede execution. Existing true-ASSERT resumption/proof rules remain unchanged.
* `nvm2c_emit` routes required ownership exclusively through a successful complete
  `nvm_verify_owned_module`, then direct owned emission. No ordinary fallback is
  introduced. The new native carrier and opcode implementation must qualify at
  the same source pin as this public admission companion.

My test matrix uses real owner construction/consumption and operand-stack float
values, with integer loop counters. I require all four public APIs, repeated
normal/assertion-failure cleanup, strict native sanitizers and the existing
reference/value-graph/result gates. I add explicit checked refusal controls for
FLOAT locals, parameters/results and owner fields without executing refused
modules. I do not reuse any previously refused executable artifact. Parent review
of this contract and the bounded whitelist delta precedes any fresh execution.
