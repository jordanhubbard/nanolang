# NanoCore Formal Specification

*Auto-generated from Coq proof scripts on 2026-02-08.*

> **NanoCore** is the formally verified subset of NanoLang. Every property
> listed below has been machine-checked in Coq with **0 axioms** and
> **0 Admitted** -- the proofs are complete and trustworthy.

## Verification Status

| Property | File | Description |
|----------|------|-------------|
| Type Soundness | `Soundness.v` | Well-typed programs evaluate to well-typed values |
| Progress | `Progress.v` | Well-typed expressions are never stuck |
| Determinism | `Determinism.v` | Evaluation produces a unique result |
| Big-step / Small-step Equivalence | `Equivalence.v` | Both semantics agree on all programs |

## 1. Types

NanoCore has **7 types** (defined in `Syntax.v`):

| Type | Syntax | Description |
|------|--------|-------------|
| `TInt` | `int` | 64-bit signed integer |
| `TBool` | `bool` | Boolean (true/false) |
| `TString` | `string` | Immutable string |
| `TUnit` | `unit` | Unit type (void-like) |
| `TArrow` | `T1 -> T2` | Function type |
| `TArray` | `array<T>` | Homogeneous array |
| `TRecord` | `{f1: T1, ..., fn: Tn}` | Record (struct) type |

## 2. Operators

### 2.1 Binary Operators

NanoCore has **14 binary operators** classified by the semantics as:
- Arithmetic (5): `OpAdd`, `OpSub`, `OpMul`, `OpDiv`, `OpMod`
- Comparison (6): `OpEq`, `OpNe`, `OpLt`, `OpLe`, `OpGt`, `OpGe`
- Logical (2): `OpAnd`, `OpOr`
- String (1): `OpStrCat`

| Operator | Symbol | Operand Types | Result Type | Notes |
|----------|--------|---------------|-------------|-------|
| `OpAdd` | `+` | int, int | int | Addition |
| `OpSub` | `-` | int, int | int | Subtraction |
| `OpMul` | `*` | int, int | int | Multiplication |
| `OpDiv` | `/` | int, int | int | Division (None if divisor = 0) |
| `OpMod` | `%` | int, int | int | Modulo (None if divisor = 0) |
| `OpEq` | `==` | int/bool/string | bool | Equality (polymorphic) |
| `OpNe` | `!=` | int/bool/string | bool | Inequality (polymorphic) |
| `OpLt` | `<` | int, int | bool | Less than |
| `OpLe` | `<=` | int, int | bool | Less than or equal |
| `OpGt` | `>` | int, int | bool | Greater than |
| `OpGe` | `>=` | int, int | bool | Greater than or equal |
| `OpAnd` | `and` | bool, bool | bool | Logical AND (short-circuit) |
| `OpOr` | `or` | bool, bool | bool | Logical OR (short-circuit) |
| `OpStrCat` | `+` | string, string | string | String concatenation |

### 2.2 Unary Operators

| Operator | Symbol | Operand Type | Result Type | Description |
|----------|--------|--------------|-------------|-------------|
| `OpNeg` | `- (unary)` | int | int | Arithmetic negation |
| `OpNot` | `not` | bool | bool | Logical negation |
| `OpStrLen` | `str_length` | string | int | String length |
| `OpArrayLen` | `array_length` | array<T> | int | Array length |

### 2.3 Arithmetic Semantics (`eval_arith_binop`)

```
eval_arith_binop op n1 n2 =
  | OpAdd => Some (VInt (n1 + n2))
  | OpSub => Some (VInt (n1 - n2))
  | OpMul => Some (VInt (n1 * n2))
  | OpDiv => if Z.eqb n2 0 then None else Some (VInt (Z.div n1 n2))
  | OpMod => if Z.eqb n2 0 then None else Some (VInt (Z.modulo n1 n2))
  | _ => None
```

### 2.4 Comparison Semantics (`eval_cmp_binop`)

```
eval_cmp_binop op n1 n2 =
  | OpEq => Some (VBool (Z.eqb n1 n2))
  | OpNe => Some (VBool (negb (Z.eqb n1 n2)))
  | OpLt => Some (VBool (Z.ltb n1 n2))
  | OpLe => Some (VBool (Z.leb n1 n2))
  | OpGt => Some (VBool (Z.ltb n2 n1))
  | OpGe => Some (VBool (Z.leb n2 n1))
  | _ => None
```

## 3. Expression Forms

NanoCore has **16 expression forms** (defined in `Syntax.v`):

| # | Constructor | Coq Comment | Description |
|---|-------------|-------------|-------------|
| 1 | `EInt` | integer literal | |
| 2 | `EBool` | boolean literal | |
| 3 | `EString` | string literal | |
| 4 | `EUnit` | unit literal | |
| 5 | `EVar` | variable reference | |
| 6 | `EBinOp` | binary operation | |
| 7 | `EUnOp` | unary operation | |
| 8 | `EIf` | if cond then e1 else e2 | |
| 9 | `ELet` | let x = e1 in e2 | |
| 10 | `ESet` | set x = e | |
| 11 | `ESeq` | e1; e2 | |
| 12 | `EWhile` | while cond do body | |
| 13 | `ELam` | fun (x : T) => body | |
| 14 | `EApp` | function application | |
| 15 | `EFix` | fix f (x:T1):T2 = body | |
| 16 | `EArray` |  | |

## 4. Evaluation Rules

The big-step semantics relation is `eval renv e renv' v`: expression `e`
evaluates to value `v` in environment `renv`, producing updated environment `renv'`.

**Key properties:**
- **Eager** (call-by-value): arguments are evaluated before function bodies
- **Deterministic**: each expression has at most one result
- **Lexical scoping**: closures capture their creation environment
- **Store-passing**: mutable variables are threaded through the environment

### 4.1 Literals

| Rule | Description |
|------|-------------|
| `E_Int` | Integer literal |
| `E_Bool` | Boolean literal |
| `E_String` | String literal |
| `E_Unit` | Unit literal |

### 4.2 Variables

| Rule | Description |
|------|-------------|
| `E_Var` | Variable lookup |

### 4.3 Arithmetic

| Rule | Description |
|------|-------------|
| `E_BinArith` | Arithmetic binary operations: int op int -> int |

### 4.4 Comparison

| Rule | Description |
|------|-------------|
| `E_BinCmp` | Comparison operations: int op int -> bool |
| `E_BinEqBool` | Equality on booleans: bool op bool -> bool |
| `E_BinNeBool` | Inequality on booleans |
| `E_BinEqStr` | String equality |
| `E_BinNeStr` | String inequality |

### 4.5 String Ops

| Rule | Description |
|------|-------------|
| `E_StrCat` | String concatenation |

### 4.6 Logical Ops

| Rule | Description |
|------|-------------|
| `E_And_True` | Logical AND (short-circuit) |
| `E_And_False` | AND where left is false (literal false on right) |
| `E_And_Short` | Short-circuit AND: if left is false, result is false regardless of right |
| `E_Or_False` | Logical OR (short-circuit) |
| `E_Or_Short` | Short-circuit OR: if left is true, result is true regardless of right |

### 4.7 Unary Ops

| Rule | Description |
|------|-------------|
| `E_Neg` | Unary negation: -n |
| `E_Not` | Logical not |
| `E_StrLen` | String length |

### 4.8 Control Flow

| Rule | Description |
|------|-------------|
| `E_IfTrue` | If-then-else: true branch |
| `E_IfFalse` | If-then-else: false branch |
| `E_Let` | Let binding: let x = e1 in e2. The body evaluates in the extended environment. The output environment pops the let binding, preserving any mutations to variables in the enclosing scope. |
| `E_Set` | Set: mutable variable update |
| `E_Seq` | Sequence: e1; e2 |
| `E_WhileTrue` | While loop: condition true, execute body and loop |
| `E_WhileFalse` | While loop: condition false, stop |

### 4.9 Functions

| Rule | Description |
|------|-------------|
| `E_Lam` | Lambda abstraction: creates a closure |
| `E_App` | Function application. The body evaluates in the closure's environment (lexical scoping). Mutations inside the body don't affect the caller's environment. The output environment is the caller's env after evaluating the function and argument. |
| `E_Fix` | Fix: recursive function creates a fix closure |
| `E_AppFix` | Application of fix closure: unrolls one step |

### 4.10 Arrays

| Rule | Description |
|------|-------------|
| `E_ArrayNil` | Array literal: empty |
| `E_ArrayCons` | Array literal: evaluate elements left-to-right |
| `E_Index` | Array indexing: (at arr i) |
| `E_ArrayLen` | Array length |
| `E_ArraySet` | Array functional update |
| `E_ArrayPush` | Array push |

### 4.11 Records

| Rule | Description |
|------|-------------|
| `E_RecordNil` | Record literal: empty |
| `E_RecordCons` | Record literal: evaluate fields left-to-right |
| `E_Field` | Record field access |
| `E_SetField` | Record field update: set x.f = e |

### 4.12 Variants

| Rule | Description |
|------|-------------|
| `E_Construct` | Variant constructor |
| `E_Match` | Pattern matching |

### 4.13 String Indexing

| Rule | Description |
|------|-------------|
| `E_StrIndex` | String indexing: total, out-of-bounds returns "" |

## 5. Typing Rules

The typing judgment is `has_type ctx e t`: expression `e` has type `t`
in typing context `ctx`.

### 5.1 Literals

| Rule | Description |
|------|-------------|
| `T_Int` | Integer literal |
| `T_Bool` | Boolean literal |
| `T_String` | String literal |
| `T_Unit` | Unit literal |

### 5.2 Variables

| Rule | Description |
|------|-------------|
| `T_Var` | Variable |

### 5.3 Operators

| Rule | Description |
|------|-------------|
| `T_BinOp` | Binary operator on integers |
| `T_BinLogic` | Logical operators on booleans |
| `T_BinEqBool` | Equality/inequality on booleans |
| `T_StrCat` | String concatenation |
| `T_BinEqStr` | Equality/inequality on strings |
| `T_Neg` | Unary negation |
| `T_Not` | Logical not |
| `T_StrLen` | String length |

### 5.4 Control Flow

| Rule | Description |
|------|-------------|
| `T_If` | If-then-else |
| `T_Let` | Let binding |
| `T_Set` | Set: mutable variable assignment |
| `T_Seq` | Sequence |
| `T_While` | While loop |

### 5.5 Functions

| Rule | Description |
|------|-------------|
| `T_Lam` | Lambda abstraction |
| `T_App` | Function application |
| `T_Fix` | Recursive function |

### 5.6 Arrays

| Rule | Description |
|------|-------------|
| `T_ArrayNil` | Empty array literal: [] has type array<T> for any T |
| `T_ArrayCons` | Non-empty array literal: [e, ...es] |
| `T_Index` | Array indexing: (at arr i) |
| `T_ArrayLen` | Array length |
| `T_ArraySet` | Array functional update |
| `T_ArrayPush` | Array push |

### 5.7 Records

| Rule | Description |
|------|-------------|
| `T_RecordNil` | Empty record literal |
| `T_RecordCons` | Non-empty record literal |
| `T_Field` | Record field access |
| `T_SetField` | Record field update |

### 5.8 Variants

| Rule | Description |
|------|-------------|
| `T_Construct` | Variant constructor |
| `T_Match` | Pattern matching |

### 5.9 Strings

| Rule | Description |
|------|-------------|
| `T_StrIndex` | String indexing |

## 6. Values

Values are results of evaluation that cannot be reduced further.
NanoCore has **9 value forms** (defined in `Syntax.v`):

| Constructor | Form | Description |
|-------------|------|-------------|
| `VInt` | `VInt n` | Integer value |
| `VBool` | `VBool b` | Boolean value |
| `VString` | `VString s` | String value |
| `VUnit` | `VUnit` | Unit value |
| `VClos` | `VClos x body env` | Closure: captures parameter name, body, and environment |
| `VFixClos` | `VFixClos f x body env` | Recursive closure: also captures the function name for self-reference |
| `VArray` | `VArray vs` | Array of values |
| `VRecord` | `VRecord fields` | Record of named field values |
| `VConstruct` | `VConstruct tag v` | Variant value: a tag with a payload value |

## 7. Edge Cases

These behaviors are precisely specified by the formal semantics:

| Case | Expression | Behavior |
|------|-----------|----------|
| Division by zero | `a / 0` and `a % 0` | Returns `None` (evaluation gets stuck). Programs must guard against zero divisors. |
| Array out of bounds | `arr[i]` where `i >= length(arr)` or `i < 0` | Returns `None` (evaluation gets stuck). The type system does not prevent this. |
| Undefined variable | `x` where `x` is not in the environment | Returns `None` (evaluation gets stuck). The type system prevents this for well-typed programs. |
| Type mismatch at runtime | e.g. `3 + true` | Gets stuck (no evaluation rule applies). Well-typed programs never reach this state (progress theorem). |
| Short-circuit AND | `false and e2` | Returns `false` without evaluating `e2`. Side effects in `e2` are **not** executed. |
| Short-circuit OR | `true or e2` | Returns `true` without evaluating `e2`. Side effects in `e2` are **not** executed. |
| String indexing out of bounds | `s[i]` where `i >= length(s)` or `i < 0` | Returns `""` (empty string). String indexing is **total** -- it never gets stuck. |
| Empty array literal | `[]` | Has type `array<T>` for any `T`. Type is determined by usage context. |
| Let binding scoping | `let x = e1 in e2` | After `e2` evaluates, the `x` binding is popped. Mutations to outer variables are preserved. |
| Closure capture | `fun (x : T) => body` | Captures the current environment (lexical scoping). Mutations inside the body do not affect the caller. |
| While loop result | `while cond do body` | Always produces `unit`. Termination is **not** guaranteed -- depends on the condition. |
| Record field update | `set x.f = v` | Updates the field in-place. Requires `x` in scope and `f` to be an existing field. |
| Pattern matching scope | `match e { Tag x => body, ... }` | The binding `x` is scoped to the match arm body and popped afterwards. |

## 8. Proven Properties

### 8.1 Type Soundness (`Soundness.v`)

If expression `e` has type `t` in a well-formed context, and `e` evaluates
to value `v`, then `v` is a well-typed value of type `t`. In other words,
evaluation preserves types.

### 8.2 Progress (`Progress.v`)

If expression `e` has type `t`, then either `e` is already a value or `e` can
take a step under the small-step semantics. Well-typed programs never get stuck
(modulo division by zero and out-of-bounds array access, which are partial).

### 8.3 Determinism (`Determinism.v`)

If `eval renv e renv1 v1` and `eval renv e renv2 v2`, then `v1 = v2` and
`renv1 = renv2`. Evaluation is a partial function -- there is at most one
result for any expression in any environment.

### 8.4 Big-step / Small-step Equivalence (`Equivalence.v`)

The big-step semantics (natural semantics) and the small-step semantics
(structural operational semantics) agree on all programs. This gives two
independent definitions of NanoCore's behavior that serve as a cross-check.

## 9. NanoCore vs Full NanoLang

NanoCore is a verified subset. Features present in full NanoLang but
**outside** the formally verified subset:

| Feature | NanoLang | NanoCore |
|---------|----------|----------|
| Integers, Booleans, Strings | Yes | Yes |
| Arrays, Records, Variants | Yes | Yes |
| Pattern Matching | Yes | Yes |
| Functions, Recursion | Yes | Yes |
| Mutable Variables | Yes | Yes |
| While Loops | Yes | Yes |
| Floats | Yes | **No** |
| Tuples | Yes | **No** |
| Hashmaps | Yes | **No** |
| Enums | Yes | **No** |
| Generics | Yes | **No** |
| Modules / Imports | Yes | **No** |
| FFI / Extern | Yes | **No** |
| Opaque Types | Yes | **No** |
| For Loops | Yes | **No** (expressible as while + let) |
| Print / Assert | Yes | **No** |

---

*Generated by `tools/extract_spec.py` from the Coq sources in `formal/`.*
