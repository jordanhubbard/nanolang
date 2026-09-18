# My bit-observable binary64 arithmetic policy prerequisite

I record `task_50090616040044dfa3bba1ab93a9f6d1` under reconstruction parent
`task_4bd034f6029b7458201db74e2c3aeb32` at main `29ad4b9f`, before production.
This is a policy proposal requiring independent review, not an implemented
semantic change or an arithmetic reconstruction admission.

## My static evidence

- `spec/nanoisa.yaml` describes f64.add/sub/mul/div and stack types, but does not
  select arithmetic NaN result bits or state a complete rounding environment.
- `src/nanovm/vm.c` typed dispatch uses ordinary C double arithmetic. DIV tests
  either signed-zero divisor first and returns positive zero even for a NaN or
  infinity numerator. Generic scalar arithmetic uses the same host operations
  after existing int/enum promotion; those promotion/tag rules are separate.
- `src/nanoisa/nvm2c.c` emits ordinary float operators and the same zero-divisor
  conditional. Concrete and boxed generic numeric helpers are additional sites
  that must follow any adopted scalar policy; typed negation is separate.
- `src/nanoisa/nvm2llvm.c` emits plain fadd/fsub/fmul, guarded float_divide and
  fneg. Generic numeric operations share floating arithmetic helpers. The Wasm
  product consumes this LLVM lowering; it has no independent source operator
  implementation that would establish a stronger NaN-payload promise.
- `src/eval.c` scalar FLOAT/FLOAT arithmetic uses native double operations and
  the positive-zero division guard. Specialized evaluation and array arithmetic
  paths also contain host operators; they need an explicit inventory before a
  language-wide policy claim, not accidental coverage through a scalar patch.
- `src/transpiler_iterative_v3_twopass.c` emits direct binary C operators for
  ordinary float expressions. `src_nano/transpiler.nano` maps slash and other
  float operators to C operators. These legacy source products must be included
  in qualification, including total division; canonical source producers instead
  select typed F64 opcodes in `src/nanovirt/codegen.c` and
  `src_nano/compiler/nanoisa_codegen.nano`.
- Existing `tests/test_native_floats.py`, `tests/test_nvm2llvm_floats.py` and
  `tests/test_llvm_generic_numeric.py` exercise values, unordered comparisons,
  signed zeros and total division. That evidence does not specify every
  arithmetic NaN payload. Exact bit transport now makes those payloads observable.

I have not run a new mismatch probe or inferred a particular host failure from
this audit. The concrete prerequisite is an underspecified observable policy.
I preserve current F64_TO_BITS admission; I will not impose the initially
considered whole-module ban merely to avoid this decision.

## My policy options

1. I can retain host-selected NaN result bits as explicitly nondeterministic.
   This preserves current implementations but cannot support exact cross-route
   integer observations of arbitrary arithmetic results without a weaker
   reconstruction equivalence relation. It does not meet the intended exact
   bit-observable continuation.
2. I can select a fixed arithmetic NaN result encoding. This is my recommended
   bounded centralized policy: scalar binary ADD/SUB/MUL/DIV results that are
   NaN become positive quiet NaN `0x7ff8000000000000`, regardless of operand
   order, input payload/sign or the invalid operation that produced it.
3. I can specify deterministic operand-payload selection and quieting, including
   left/right priority and invalid-operation defaults. This preserves more
   diagnostic payload information but needs more branches, precise promotion
   ordering and a larger cross-backend contract. No current documented rule
   requires this complexity.

Options 2 and 3 are semantic changes to newly observable arithmetic bits, not
bug fixes that can be silently folded into reconstruction. Neither may alter
input values, PUSH_F64, FROM_BITS/TO_BITS, copies, comparison, or F64_NEG:
transport and negation retain every payload and signaling bit as qualified.

## My proposed deterministic scalar contract, pending review

I retain exact binary64 operations with round-to-nearest, ties-to-even in the
ordinary default environment, gradual underflow, signed infinities and signed
zeros. I preserve signed-zero results dictated by each operation; I normalize
only NaNs. DIV checks a signed-zero divisor before arithmetic and returns exact
positive zero even for a NaN/infinity numerator. Nonzero division and other
binary operations normalize a NaN result to the one selected encoding.

I do not permit fast-math, reassociation, intermediate excess precision or fused
multiply-add to erase separately observable operation boundaries. Implementation
must establish binary64 rounding per operation, including reconstructed source
and optimized native/LLVM paths. A shared C helper plus matched LLVM helper can
centralize policy; helper boundaries alone do not prove that contraction or
excess precision is disabled. Target/compiler guards and qualification must
establish this. I make no promise for caller-modified rounding, enabled floating
traps or flush-to-zero modes in this bounded contract.

I use integer representation checks and memcpy for result normalization. I do
not quiet or canonicalize operands before arithmetic, and do not reuse a helper
that canonicalizes transport or unary negation. Generic scalar float results
must match the policy while retaining their existing integer promotion and
invalid-tag rules. Arrays, external math functions and unsupported owned-profile
arithmetic must be explicitly inventoried and either included in a reviewed
language-wide rollout or retained as separately named obligations; I will not
claim all numeric routes from only scalar dispatch changes.

## My required qualification before arithmetic reconstruction

1. I complete the exact call-site inventory and obtain policy review before any
   implementation. I update normative schema/docs with the explicit semantic
   change and preserve source/ISA compatibility boundaries accurately.
2. I implement shared scalar behavior across VM dispatch, concrete/boxed native,
   interpreter, legacy C/selfhost products and LLVM/Wasm. Canonical producers
   retain typed opcode selection. Unsupported profiles retain checked refusal.
3. I freeze production and harness before ordinary fresh gates. Exact integer
   observers cover distinct signed qNaN/sNaN payloads in both operand positions,
   invalid infinity operations, NaN divided by both signed zeros, overflow,
   gradual underflow, halfway ties and separate-operation rounding. Original
   operand bits, transport and negation must remain unchanged.
4. I qualify unoptimized/optimized VM/C/LLVM/Wasm, Wasmtime and Node, source
   interpreter/C-seed/Stage1/Stage2 routes and GCC/Clang sanitizer controls.
   Fresh tools containing the reviewed policy are required: immutable PR720
   binaries remain comparison evidence, not silently modified qualified tools.
5. Only after this prerequisite passes do I record and implement the separate
   typed binary-arithmetic reconstruction admission. Existing bit transport and
   other reconstruction functionality remain available throughout.

I retain the first failure of any fresh gate. I preserve historical PR679 and
frozen product artifacts. Full reconstruction and release remain open.

## My selected policy and exact rollout inventory

Independent review selected option 2 after `baa865b4`: I canonicalize every
scalar binary arithmetic NaN result to `0x7ff8000000000000`. The following
inventory and staging remain subject to review before production. Line numbers
refer to my `29ad4b9f` base; symbols remain the durable source pointers.

| Route | Exact production sites | Required change or preserved boundary |
| --- | --- | --- |
| VM typed scalar | `src/nanovm/vm.c:2202`, F64_ADD/SUB/MUL/DIV group | Shared four-operation policy helper after exact tag checks |
| VM generic scalar | `src/nanovm/vm.c:1984`, `2031`, `2068`, `2112`, all FLOAT/FLOAT and mixed INT/FLOAT arms | Same helper after existing enum/int promotion; integer branches unchanged |
| Native typed | `src/nanoisa/nvm2c.c:3654`, F64 binary emission | Emit standalone policy helpers and calls; keep NEG/comparisons separate |
| Native known generic | `src/nanoisa/nvm2c.c:3695`, known FLOAT numeric branch | Same helper after existing exact conversion checks |
| Native boxed generic | `src/nanoisa/nvm2c.c:6549`, emitted nvalue_numeric | Same helper only for four binary FLOAT results; integer and unary paths unchanged |
| LLVM and Wasm typed | `src/nanoisa/nvm2llvm.c:600`, F64 emission | Four shared strict arithmetic helpers, integer-bit NaN normalization; no fast-math flags |
| LLVM and Wasm generic | `src/nanoisa/nvm2llvm.c:72`, float_divide and numeric binary helper loop | Reuse strict helpers after current tag/promotion checks; Wasm uses this IR |
| Main interpreter scalar | `src/eval.c:2736`, FLOAT/FLOAT prefix arithmetic | C policy helper; preserve operand evaluation and other tags |
| Optimized interpreter callbacks | `src/eval.c:1772` eval_pure_expr_float2 and `1806` eval_pure_expr_float | Same helper at each binary node; callers at `1970` and `2332` are map/reduce callback optimization, so these scalar operators are included |
| C-seed legacy C | `src/transpiler_iterative_v3_twopass.c:1433` and ordinary binary emission | Detect exact scalar float expressions before generic operator fallback; emit helper calls with once-evaluated operands and total division |
| Selfhost legacy C | `src_nano/transpiler.nano:5297`, generate_expression binary fallback; gen_c_runtime | Exact scalar float helper selection plus emitted standalone helpers; preserve unary/array/string paths |
| Canonical C-seed/selfhost producers | `src/nanovirt/codegen.c:2360`; `src_nano/compiler/nanoisa_codegen.nano:3374` | Existing typed opcode selection stays; inspect emitted modules, no new opcode or version |
| Owned native profile | `src/nanoisa/nvm2c_owned.h:158`, integer arithmetic only | No float admission; preserve verifier/profile refusal and exact integer implementation |
| Managed scalar consumers | Existing scalar verifier/emitter delegation in LLVM | Same scalar helper policy, no new heap/opcode admission |

Static review identifies a concrete total-division inconsistency: both optimized
interpreter float helpers return raw `a / b`, unlike the main scalar interpreter
and VM. C-seed legacy emission explicitly says float division remains plain `/`;
selfhost legacy binary fallback also renders `/`. Correcting these scalar routes
is a required prerequisite, not merely NaN canonicalization. I have not executed
an old failing case or claimed its observed impact.

## My per-operation rounding controls

My shared C implementation will require `sizeof(double)==8`, `FLT_RADIX==2`,
`DBL_MANT_DIG==53`, `DBL_MAX_EXP==1024`, `DBL_MIN_EXP==-1021` and
`FLT_EVAL_METHOD==0`. I reject unsupported storage/evaluation targets at compile
time rather than call extended-precision rounding equivalent. I reject
`__FAST_MATH__` and nonzero `__FINITE_MATH_ONLY__` in the helper contract.

Each helper computes exactly one binary operation into a volatile double local,
then loads that stored binary64 value and inspects its integer representation.
That observable volatile store/load prevents an outer operation from contracting
with this operation even if helpers inline. I also qualify C paths with explicit
`-ffp-contract=off` and `-fno-fast-math`, and inspect supported compiler controls
in generated/driver builds. A helper name or noinline attribute alone is not my
rounding argument. I preserve existing operand snapshots and do not reevaluate
operand expressions to normalize the result. A bitwise zero-divisor test can
select exact positive zero before division without touching NaN operand bits.

LLVM helpers use plain fadd/fsub/fmul/fdiv without fast, reassoc or contract flags,
followed by a bitcast and integer exponent/fraction test. Normalization selects
canonical bits; it never selects an operand's payload. A result bitcast/normalize
boundary and absence of contraction permission preserve separate operations.
Wasm lowering inherits the binary64 operations and explicit result normalization;
Node and Wasmtime both remain required qualification routes.

I require the ordinary default round-to-nearest/ties-even environment with
gradual underflow. Tests check that environment rather than silently resetting a
caller's mode. Host-changed modes/traps/flush-to-zero remain outside this scoped
contract. Direct helper tests use runtime operands and include O0/O2 plus LTO
where supported, halfway ties, minimum-subnormal halving, three-subnormal
halving, signed-zero multiplication/division, and a contraction-sensitive
`(1+2^-52)*(1-2^-52)-1` chain. Each operation rounds separately; the last example
must yield positive zero rather than a fused negative `2^-104` result.

## My precise aggregate and foreign scope

Scalar callback operators remain included even when map/reduce invokes them.
Array *operators* are separate: `vm_array_arithmetic` at vm.c:74/136, interpreter
array and broadcast branches in eval.c, C array runtime helpers selected by the
legacy emitter, and selfhost `build_array_binop_expr` loops. I record
`task_3717eda847a74122916571444478ee0f` for their explicit policy/qualification.
I do not incidentally alter those loops or claim their results are canonicalized
by this scalar rollout. External math/FFI results retain provider contracts;
merely transporting a NaN from a provider must not normalize it. A subsequent
admitted scalar binary operation normalizes its own result.

## My implementation and acceptance stages

1. I add one reviewed C scalar policy implementation and standalone emission
   mechanism, target guards and exact helper tests. I qualify the four operations,
   zero precedence, canonical result bits and untouched input/transport/negation.
   I retain independent arrays/external semantics in normative documentation.
2. I apply the helper to VM typed/generic, native typed/known/boxed, and LLVM
   typed/generic sites. Same-module exact-bit tests run VM, GCC/Clang native,
   LLVM unoptimized/optimized/native, Wasmtime and Node. Managed scalar wrappers
   and unsupported owned-profile refusal get focused controls; no heap widening.
3. I apply the same scalar policy to main/optimized interpreter and both legacy
   C emitters, repairing raw scalar division. Ordinary callback, once-evaluation,
   mutable source, exact tag and full source-producer parity gates follow. I use
   a fresh isolated bootstrap with reviewed source; immutable PR720 tools remain
   historical qualified artifacts, not rewritten inputs. Normative policy cannot
   be reported complete before this source stage passes.
4. I freeze each complete staged production/harness checkpoint before its gates,
   retain first failures and publish exact evidence for review. I do not claim a
   cross-route policy from only the first stage. Array follow-up and broader
   language/release parents remain open until their distinct acceptance passes.
5. After policy completion I return to a separate reconstruction arithmetic
   contract. Existing F64_TO_BITS, transport, negation and comparison admission
   remains intact throughout; no workaround bans are introduced.

## My emission namespace boundary

Review of helper checkpoint `b53be06d` found that nl_f64_add/sub/mul/div
would collide with legacy C names for ordinary user functions f64_add/sub/mul/div.
No product path emitted these helpers at that checkpoint. Before tests/integration
I use the established nano_rt_ runtime prefix (as existing nano_rt_idiv does).
I do not reserve ordinary f64_* source identifiers. Helper tests retain the old
nl_f64_* user symbols beside the new helper names; source integration requires
paired ordinary named-function execution through both producer implementations.
Foreign explicit ABI symbols remain subject to the existing runtime namespace
boundary and must not be inferred to be ordinary prefixed source functions.

## My stage-3 source initialization and evaluation boundary

I begin source integration on `c8db7cc5` after merged PR731. C-seed
is_c_constant_initializer accepts literals only; arithmetic already belongs to
its ordered, once-only runtime initializer. I preserve that decision.
Selfhost global_init_literal otherwise falls back to zero for arithmetic. I add
an exact FLOAT arithmetic-root case to its existing startup list for both plain
and guarded globals, preserving declaration order and the separate bit-intrinsic
case. I do not use bit-intrinsic detection as permission for arbitrary global
expressions. Paired tests cover mutable/immutable globals, references to earlier
globals, nested arithmetic and effectful operands initialized once.

I emit each scalar operand once into an ordered left/right statement-expression
snapshot before the helper call, following existing GNU C expression machinery.
C-seed fresh temporary names avoid environment-visible source names; selfhost
private temporary identifiers stay outside its nl_ source-variable mapping.
Ordinary f64_add/sub/mul/div functions remain callable alongside nano_rt_ runtime
helpers. I generate both embedded C text and a Nano runtime-string provider from
one canonical header; regeneration checks prevent independent helper drift.

## My additional public C-source target prerequisite

My stage-3 identity test exposed a missing inventory route: C-seed `--target c`
uses `src/c_backend.c`, not the native legacy emitter. Its file-level contract
advertises scalar numeric arithmetic and self-contained C output; current ordinary
source output still contains raw arithmetic and unresolved bit intrinsic calls.
I did not compile or execute that output. Task `task_070dbdb1d2e04a36b09a5eab25ddddb3`
requires exact typing/transport and arithmetic-policy qualification for this
public route. I retain parent500906 open until this prerequisite is resolved.
My native legacy helper identity test instead uses C-seed `--keep-c`; selfhost
`--target c` already selects its legacy transpiler. These route names are not
interchangeable.
