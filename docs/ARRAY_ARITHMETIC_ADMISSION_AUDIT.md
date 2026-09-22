# My array arithmetic admission audit

I audit source at `653af65401d8fde9253f5671c96ccc785fae340e` after its
full C checker/evaluator gates pass. This is a source audit and proposed
contract, not new execution evidence. I execute no known unsupported route.
Task `task_398942bdc317a4a778f70f1245ddc047` remains open under task992.

## My existing acceptance boundary

`NANOISA_AGGREGATE_BINARY64_POLICY.md` requires existing flat int/float
array pairs and both matching scalar broadcasts for + - * /. Its exact
selfhost classifier deliberately excludes bytes and nested arrays. The
contract preserves recursive interpreter/C-helper leaf behavior without
claiming nested source admission. `EMITTED_ARRAY_BINARY64_QUALIFICATION.md`
likewise tests recursive helper leaves, not additional source shapes.

The source fixtures `tests/test_aggregate_binary64_policy.py` exercise flat
FLOAT pairs and broadcasts, operand order, empties, INT neighbors and mixed
refusals. `tests/test_array_operators.nano` additionally retains legacy STRING
addition. Neither fixture establishes byte-array or nested-array arithmetic.
Byte array mutation, literals, indexing and slicing tests are separate;
I must preserve them and cannot count them as arithmetic acceptance.

`NANOISA_ONLY.md` still requires same-module VM/C/LLVM/Wasm semantic
qualification for the selected full5.1 scope. A checked capability refusal
is a safety boundary, not a substitute for a required positive matrix cell.
These findings do not close the aggregate parent or release gate.

## What my implementations actually select

| Layer | Static selection and gap |
| --- | --- |
| C coarse checker (`typechecker.c`, arithmetic AST_PREFIX_OP) | Unary ARRAY accepts unknown/int/enum/float leaves. Binary accepts unknown array leaves early and normalizes enum/u8 to int. Known nested leaves fail. A standalone expression can therefore pass without a complete destination view. |
| C complete view (`typechecker_nominal_arrays.inc`, `nominal_array_arithmetic`) | Qualified653 derives flat int/enum-to-int, float and STRING+ results; scalar u8 promotes to int. Actual byte and nested arrays have no result view. This alone does not protect discarded expressions. |
| Evaluator (`eval.c`, `eval_dyn_array_*`, unary branch) | Flat ELEM_INT/FLOAT arithmetic and STRING+ exist. Binary nested helpers recurse; unary does not. ELEM_U8 numeric dispatch is absent. Enum source values have an integer carrier. Fixed/dynamic array length/shape behavior remains existing policy. |
| C legacy native (`transpiler_iterative_v3_twopass.c`, array operator branches) | Typed byte loops read/push ELEM_U8 and retain byte result storage, unlike scalar promotion. Broadcast temporaries use uint8_t too. Nested pair fallback recurses into runtime helpers. Unary supports int/float only. These snippets are not qualified byte/nested source behavior. |
| Nano checker (`typecheck.nano`, `array_arithmetic_*`, `check_binary_op`, `check_expr_node`) | Exact flat INT/FLOAT profiles and matching scalar kinds are selected for four binary operators. Byte/nested/string/modulo fall into older paths; TYPE_UNKNOWN can escape without this profile's diagnostic. The binary-node checker does not separately select `is_unary`. |
| Nano canonical (`compiler/nanoisa_codegen.nano`, `nisa_array_arithmetic_type`, emission) | Exact array<int>/int or array<float>/float four-op profile. Arithmetic arrays outside it explicitly fail. Unary accepts scalar int/enum/u8/float only. No byte/nested array admission. |
| C NanoISA producer (`nanovirt/codegen.c`, AST_PREFIX_OP) | Binary array +-*/ selects ARRAY_ADD/SUB/MUL/DIV. Unary ARRAY follows scalar I64_NEG selection; ARRAY remainder follows scalar I64_REM_S. These are static operand/opcode mismatches requiring refusal or separately reviewed lowering. No invalid module was executed in this audit. |
| VM (`nanovm/vm.c`, `vm_array_arithmetic`) | Flat INT/FLOAT and STRING+ element pairs; mixed numeric tags promote to FLOAT. Byte and nested element tags fail classification. It keeps a shorter-array rule, unlike legacy equal-length helpers. Enum scalar normalization does not normalize array leaves. |
| Canonical native translators | `nvm2c` generic tagged arithmetic emits `nvalue_numeric`, whose carriers are numeric scalar kinds; `nvm2llvm` has its own scalar/managed profile checks. I found no native byte/nested arithmetic contract in the reviewed aggregate documents. Existing managed array storage support does not establish arithmetic. I do not infer a new accepted profile or a complete refusal proof from opcode searches. |

## My proposed bounded contract

I separate source semantics from target capability without changing either by
accident. Existing admitted flat evaluator semantics remain: integer operations
produce INT, enum identity is removed, scalar u8 promotes to INT; FLOAT keeps
its exact existing helper policy; STRING supports addition only. I preserve
legacy positive fixtures and all exact native/canonical profile checks.

For actual byte arrays and nested arrays, I propose an explicit unsupported
source diagnostic at arithmetic checking in every expression context, before
operand execution or output publication. Unknown/incomplete element facts may
not make an unsupported array arithmetic expression silently valid. I use the
actual resolved complete operand annotation, including declared-call and
imported/lexical results; an unrelated scalar or destination cannot supply a
missing leaf. This proposal does not add byte-array promotion, wrapping byte
results, nested recursive source arithmetic, mixed numeric source promotion,
or a new shape/length policy. Existing low-level recursive helpers stay intact.

Target lowering independently checks capability: the C NanoISA producer must
not place ARRAY operands under typed scalar NEG/REM. Where a target lacks the
already supported evaluator operation, it refuses before publication; a future
positive lowering is a separate reviewed matrix obligation. In particular, I
do not remove the passing original unary evaluator tests merely because the
canonical producer lacks their lowering. Legacy byte loops cannot be used as
an accidental alternative semantic definition.

I must review this proposal before implementation. A narrow shared-classifier
change needs original checker positives plus negative checker diagnostics for
both operands, nested wrappers, discarded expressions, let/return/call contexts,
actual imported and lexical identities, and every arithmetic operator. Compiler
negative tests check diagnostic and unchanged output sentinel without executing
an unsupported product. Existing flat aggregate source/helper fixtures and
original full checker/evaluator gates remain required. Any change to Nano
sources requires fresh producers/bootstrap under original deadlines. Full
required producer/backend positive cells remain open where capability is absent.
