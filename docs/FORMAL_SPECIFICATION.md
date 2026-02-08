# NanoCore Formal Specification

*Auto-generated from the Coq formal proofs in `formal/`.*
*This document describes the formally verified subset of NanoLang.*

## Verification Status

All properties below are proven **axiom-free** in Coq (~5,635 LOC):

| Property | File | Status |
|----------|------|--------|
| Type Soundness | `Soundness.v` | Proven |
| Progress | `Progress.v` | Proven |
| Determinism | `Determinism.v` | Proven |
| Big-step/Small-step Equivalence | `Equivalence.v` | Proven |
| Computable Evaluator Soundness | `EvalFn.v` | Partial (10/25 cases) |

## Types

NanoCore has 8 types:

| Coq Name | NanoLang Syntax | Description |
|----------|----------------|-------------|
| `TInt` | `int` |  |
| `TBool` | `bool` |  |
| `TString` | `string` |  |
| `TUnit` | `unit` |  |
| `TArrow` | `T1 -> T2` |  |
| `TArray` | `array<T>` |  |
| `TRecord` | `{f1: T1, ..., fn: Tn}` |  |

## Expression Forms

NanoCore has 16 expression forms:

| # | Coq Constructor | Description |
|---|----------------|-------------|
| 1 | `EInt` | integer literal |
| 2 | `EBool` | boolean literal |
| 3 | `EString` | string literal |
| 4 | `EUnit` | unit literal |
| 5 | `EVar` | variable reference |
| 6 | `EBinOp` | binary operation |
| 7 | `EUnOp` | unary operation |
| 8 | `EIf` | if-then-else |
| 9 | `ELet` | let binding |
| 10 | `ESet` | mutable assignment |
| 11 | `ESeq` | sequence (e1; e2) |
| 12 | `EWhile` | while loop |
| 13 | `ELam` | lambda abstraction |
| 14 | `EApp` | function application |
| 15 | `EFix` | recursive function (fix) |
| 16 | `EArray` | array literal |

## Operator Semantics

### Binary Operators

| Operator | Symbol | Operand Types | Result Type | Notes |
|----------|--------|--------------|-------------|-------|
| `OpAdd` | `+` | int, int | int | |
| `OpSub` | `-` | int, int | int | |
| `OpMul` | `*` | int, int | int | |
| `OpDiv` | `/` | int, int | int | Returns None if divisor = 0 |
| `OpMod` | `%` | int, int | int | Returns None if divisor = 0 |
| `OpEq` | `==` | int/bool/string | bool | Polymorphic equality |
| `OpNe` | `!=` | int/bool/string | bool | Polymorphic equality |
| `OpLt` | `<` | int, int | bool | |
| `OpLe` | `<=` | int, int | bool | |
| `OpGt` | `>` | int, int | bool | |
| `OpGe` | `>=` | int, int | bool | |
| `OpAnd` | `and` | bool, bool | bool | Short-circuit evaluation |
| `OpOr` | `or` | bool, bool | bool | Short-circuit evaluation |
| `OpStrCat` | `+` | string, string | string | String concatenation |

### Unary Operators

| Operator | Symbol | Operand Type | Result Type |
|----------|--------|-------------|-------------|
| `OpNeg` | `- (unary)` | int | int |
| `OpNot` | `not` | bool | bool |
| `OpStrLen` | `str_length` | string | int |
| `OpArrayLen` | `array_length` | array<T> | int |

## Evaluation Rules

The big-step semantics (`eval renv e renv' v`) evaluates expression `e`
in environment `renv`, producing value `v` and updated environment `renv'`.

Key properties:
- **Eager evaluation** (call-by-value)
- **Deterministic** (each expression has at most one value)
- **Lexical scoping** via closures
- **Store-passing** style for mutable variables

| Rule | Description |
|------|-------------|
| `E_Int` | Integer literal |
| `E_Bool` | Boolean literal |
| `E_String` | String literal |
| `E_Unit` | Unit literal |
| `E_Var` | Variable lookup |
| `E_BinArith` | Arithmetic binary operations: int op int -> int |
| `E_BinCmp` | Comparison operations: int op int -> bool |
| `E_BinEqBool` | Equality on booleans: bool op bool -> bool |
| `E_BinNeBool` | Inequality on booleans |
| `E_StrCat` | String concatenation |
| `E_BinEqStr` | String equality |
| `E_BinNeStr` | String inequality |
| `E_And_True` | Logical AND (short-circuit) |
| `E_And_Short` | Short-circuit AND: if left is false, result is false regardless of right |
| `E_Or_False` | Logical OR (short-circuit) |
| `E_Neg` | Unary negation: -n |
| `E_Not` | Logical not |
| `E_StrLen` | String length |
| `E_IfTrue` | If-then-else: true branch |
| `E_IfFalse` | If-then-else: false branch |
| `E_Set` | Set: mutable variable update |
| `E_Seq` | Sequence: e1; e2 |
| `E_WhileTrue` | While loop: condition true, execute body and loop |
| `E_WhileFalse` | While loop: condition false, stop |
| `E_Lam` | Lambda abstraction: creates a closure |
| `E_ArrayNil` | Array literal: empty |
| `E_ArrayCons` | Array literal: evaluate elements left-to-right |
| `E_Index` | Array indexing: (at arr i) |
| `E_ArrayLen` | Array length |
| `E_RecordNil` | Record literal: empty |
| `E_RecordCons` | Record literal: evaluate fields left-to-right |
| `E_Field` | Record field access |
| `E_SetField` | Record field update: set x.f = e |
| `E_Fix` | Fix: recursive function creates a fix closure |
| `E_AppFix` | Application of fix closure: unrolls one step |
| `E_Construct` | Variant constructor |
| `E_Match` | Pattern matching |
| `E_ArraySet` | Array functional update |
| `E_ArrayPush` | Array push |
| `E_StrIndex` | String indexing: total, out-of-bounds returns "" |

## Type Rules

The typing judgment (`has_type ctx e t`) assigns type `t` to expression `e`
in typing context `ctx`.

| Rule | Description |
|------|-------------|
| `T_Int` | Integer literal |
| `T_Bool` | Boolean literal |
| `T_String` | String literal |
| `T_Unit` | Unit literal |
| `T_Var` | Variable |
| `T_BinOp` | Binary operator on integers |
| `T_BinLogic` | Logical operators on booleans |
| `T_BinEqBool` | Equality/inequality on booleans |
| `T_StrCat` | String concatenation |
| `T_BinEqStr` | Equality/inequality on strings |
| `T_Neg` | Unary negation |
| `T_Not` | Logical not |
| `T_StrLen` | String length |
| `T_If` | If-then-else |
| `T_Let` | Let binding |
| `T_Set` | Set: mutable variable assignment |
| `T_Seq` | Sequence |
| `T_While` | While loop |
| `T_Lam` | Lambda abstraction |
| `T_App` | Function application |
| `T_Fix` | Recursive function |
| `T_ArrayNil` | Empty array literal: [] has type array<T> for any T |
| `T_ArrayCons` | Non-empty array literal: [e, ...es] |
| `T_Index` | Array indexing: (at arr i) |
| `T_ArrayLen` | Array length |
| `T_ArraySet` | Array functional update |
| `T_ArrayPush` | Array push |
| `T_RecordNil` | Empty record literal |
| `T_RecordCons` | Non-empty record literal |
| `T_Field` | Record field access |
| `T_SetField` | Record field update |
| `T_Construct` | Variant constructor |
| `T_Match` | Pattern matching |
| `T_StrIndex` | String indexing |

## Edge Cases

These behaviors are specified by the formal semantics and proven correct:

| Case | Expression | Behavior |
|------|-----------|----------|
| Division by zero | `a / 0` and `a % 0` | Returns `None` (evaluation gets stuck). Programs must guard against zero divisors. |
| Array out of bounds | `arr[i]` where `i >= length(arr)` or `i < 0` | Returns `None` (evaluation gets stuck). The type system does not prevent this. |
| Short-circuit AND | `false and e2` | Returns `false` without evaluating `e2`. Side effects in `e2` are not executed. |
| Short-circuit OR | `true or e2` | Returns `true` without evaluating `e2`. Side effects in `e2` are not executed. |
| String indexing | `s[i]` where `i >= length(s)` or `i < 0` | Returns `""` (empty string). String indexing is total — it never gets stuck. |
| Empty array literal | `[]` | Has type `array<T>` for any `T`. Type is determined by context. |
| Let binding scoping | `let x = e1 in e2` | After `e2` evaluates, the `x` binding is popped. Mutations to outer variables are preserved. |
| Closure capture | `fun (x : T) => body` | Captures the current environment at closure creation time (lexical scoping). |
| While loop value | `while cond do body` | Always returns `unit`. Loop value is discarded each iteration. |
| Record field update | `set x.f = v` | Updates the field in-place. Requires `x` in scope and `f` is a valid field with a prior value. |
| Pattern matching | `match e { Tag x => body, ... }` | Binding `x` is scoped to the match arm body. After the body evaluates, the binding is popped. |

## Proven Properties

### Type Soundness (Soundness.v)

If expression `e` has type `t` in context `ctx`, and `e` evaluates to value `v`,
then `v` has type `t`. Types are preserved through evaluation.

### Progress (Progress.v)

If expression `e` has type `t`, then either `e` is a value, or `e` can take
a step (small-step semantics). Well-typed programs don't get stuck.

### Determinism (Determinism.v)

If `eval renv e renv1 v1` and `eval renv e renv2 v2`, then `v1 = v2` and
`renv1 = renv2`. Evaluation is a partial function.

### Big-step/Small-step Equivalence (Equivalence.v)

The big-step semantics (natural semantics) and the small-step semantics
(structural operational semantics) agree on all programs. This provides
confidence that the specification is consistent.

### Computable Evaluator (EvalFn.v)

A fuel-based computable evaluator `eval_fn` is provided with partial soundness
proofs. If `eval_fn fuel renv e = Some (renv', v)`, then `eval renv e renv' v`.
The evaluator can be extracted to OCaml for use as a reference interpreter.

## NanoCore vs Full NanoLang

NanoCore is a subset of NanoLang. The following features are in NanoLang
but **not** in the formally verified NanoCore subset:

| Feature | NanoLang | NanoCore |
|---------|---------|----------|
| Integers | Yes | Yes |
| Booleans | Yes | Yes |
| Strings | Yes | Yes |
| Floats | Yes | **No** |
| Arrays | Yes | Yes |
| Structs/Records | Yes | Yes |
| Unions/Variants | Yes | Yes |
| Pattern Matching | Yes | Yes |
| Tuples | Yes | **No** |
| Hashmaps | Yes | **No** |
| Enums | Yes | **No** |
| Generics | Yes | **No** |
| Modules/Imports | Yes | **No** |
| FFI/Extern | Yes | **No** |
| Opaque Types | Yes | **No** |
| Unsafe Blocks | Yes | **No** |
| For Loops | Yes | **No** (equivalent to while+let) |
| Print/Assert | Yes | **No** |

Use `nanoc --trust-report <file.nano>` to see which functions in your
program fall within the verified NanoCore subset.
