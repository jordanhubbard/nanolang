# My typed scalar C expressions

I develop this private emission primitive under #1142. My existing C-text component spells byte literals, integer constants and generated names. I next select C expressions for typed scalar operations. This depends only on that component and schema-generated primary metadata; it does not depend on the blocked affine private/public walk. Complete representation, body/profile admission, lifetime cleanup, runtime emission and product cutover remain required in their existing order.

## My boundary

I add `src_nano/nvm2c/scalar_c.nano`. `emit(opcode, inputs)` takes a primary opcode and an array of generated local-register ordinals, not source names or arbitrary C fragments. I return `Expression { valid, arity, boolean, text }`. Refusal returns false, zero, false and empty text together. I accept exactly the required arity, and every supplied ordinal must pass the existing C-text local-name boundary,0 through4294967295. Repeated ordinals remain valid aliases. I use the existing `nano_aot_local_` namespace without changing the name-kind API or its old refusals.

I initially implement these fifteen typed operations:

| Family | Primary encodings | Result |
| --- | --- | --- |
| Signed wrapping arithmetic | I64_ADD192, I64_SUB193, I64_MUL194, I64_DIV_S195, I64_REM_S196, I64_NEG197 | integer |
| Signed comparisons | I64_EQ208, I64_NE209, I64_LT_S210, I64_LE_S211, I64_GT_S212, I64_GE_S213 | boolean |
| Boolean operations | BOOL_AND224, BOOL_OR225, BOOL_NOT226 | boolean |

I obtain arity from my generated ISA metadata after the explicit emission-family check. Every other primary encoding and out-of-range opcode refuses this primitive. That refusal is not a new executable profile or a decision to omit another required operation from5.1. Generic arithmetic, unsigned/bitwise/wide operations, binary64, strings, control flow, calls, aggregates, ownership and host services retain their separate implementation and acceptance requirements.

My caller owns decoded operands, checked types and register allocation. It binds already evaluated values before using my expression, in the required source/instruction order. Boolean inputs are checked booleans. Typed binary integer enum coercion remains at its existing verified boundary; I do not make unary integer or boolean operations accept enum or arbitrary tagged inputs. My result-kind flag describes the expression; it grants no wire, module, representation, callable, ownership, host or executable authority.

## My integer and boolean semantics

I use unsigned64 intermediates for addition, subtraction, multiplication and negation. My emitted fixed prelude reconstructs signed values by a range check and subtraction, avoiding out-of-range signed conversion and signed overflow. It contains ordinary C scalar code emitted by NanoLang, with no translator call or execution engine.

I preserve the current NanoVM and C-reference rules: signed division by zero returns0, INT64_MIN divided by-1 returns INT64_MIN, and remainder by zero or at that overflow pair returns0. All other division truncates toward zero; remainder follows it. I guard the exceptional branches before any C division or remainder can execute. Signed negation wraps INT64_MIN to itself.

Comparisons produce boolean0/1. After validating both ordinals, I reduce a signed comparison of the same bound register to its exact boolean, retaining a void reference to that register and arity2. This preserves aliases under strict C self-comparison diagnostics. Boolean conjunction/disjunction use only already bound values; the later body emitter must preserve eager operand evaluation before reaching this primitive. Generated register references contain no calls, source identifiers or user-provided expressions. This primitive alone does not prove whole-program evaluation order or dynamic tag correctness.

I audit these rules against `spec/nanoisa.yaml`, typed NanoVM cases in `src/nanovm/vm.c`, and the read-only C reference's `emit_binop`, `emit_unop` and integer division/remainder branches in `src/nanoisa/nvm2c.c`. My Linux execution receipt below checks these arithmetic observations against both actual engines. I edit no C product source.

## My qualification contract

I add useful shadows for arity/result classification, generated-name bounds and aliasing, unsupported and out-of-range opcodes, exact refusal fields and representative expression/prelude spelling. I preserve every old C-text assertion, including invalid name-kind6.

My permanent driver reports the prelude and complete expression results across all256 primary encodings, out-of-range opcodes, every accepted family's wrong arities, invalid/endpoint ordinals and repeated-slot inputs. I compare every report through C-seed native, C-producer VM and ASan+UBSan native, and the genuine current NanoLang producer's VM and ASan+UBSan native. I retain original600-second builds,60-second executions and ordinary ten-second shadows.

I compile the actual emitted text as strict C and execute endpoint, zero, signed-division/remainder, wrapping, boolean and deterministic additional cases against independent Python integer expectations. I also build actual NanoISA reference carriers using existing wire builders, verify them, run them in NanoVM and compile them with the reference translator. Explicit integer conversion normalizes boolean observation for this arithmetic comparison; it does not qualify representation or printing identity. Every individual normalized result must agree. I preserve original test assertions, malformed/unsupported boundaries and prior-output checks rather than using a checksum or changing an expectation after a failure.

I run the unchanged permanent C-text regression alongside the new gate. Source, fixtures, schema, tools, actual producer and immutable host/runtime libraries remain guarded. A passing independently checked and pushed receipt accepts this primitive only. Current/paired full translation, all runtime/host profiles, native compiler/translator installation, pure cutover and all69 release requirements remain open. #1250/#1263 retain their separate C-owner boundaries. The explicit #1262 handoff and [current-qualified-Stage2 Linux receipt](evidence/c-repairs-linux-20261126/current-components-20261127/README.md) now qualify its named private/public/cache/solver/callable components; complete module/profile composition remains required.


## My Linux receipt

My [complete scalar-expression receipt](evidence/darwin-current-source-20261124/diagnostics/scalar-c-expressions-five-linux-20261126/README.md) records five permanent methods and72 commands in43.078263 seconds. All415 expression reports match across five routes, with71 accepted expressions. Actual emitted C produces71,740 matching scalar observations under strict GCC ASan+UBSan/O3; a supplementary strict Clang18 O3 control produces7,174 matching observations from byte-identical retained C and inputs. Actual NanoVM and the C-reference product each match1,440 normalized observations. Both unchanged C-text methods pass, including their original oversized refusals.

I retain the missing-schema setup, the incorrect NEG classification and void-entry oracle, the strict-C alias failure, and the earlier functional pass with incomplete producer-host guards. The accepted run guards all3,975 selected source inputs,12 tool/runtime images,22 host tools, the genuine native Stage2 producer and its six loaded host libraries and bootstrap receipt before and after the original tests. No owned child or unexpected sanitizer report remains. My independent reader checks complete bytes, exact repairs and all72 recorded command arguments/budgets, then replays all five original unittest methods against the captured products without executing them again.

This accepts my private scalar-expression component on Linux. It does not complete body emission, checked profile admission, aggregate/ownership/host emission, paired-platform qualification or native cutover. My Make recipe is additive; I checked its dry run and executed its permanent methods directly under the pass-through observer. I claim no fresh bootstrap or whole-compiler speedup from this component.
