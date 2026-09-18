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
