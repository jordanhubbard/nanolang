# My Formal Specification (NanoCore)

*Auto-generated from my Coq proof scripts on 2026-02-08.*

> **NanoCore** is my formally verified subset. I have machine-checked every property
> listed below in Coq using **0 axioms** and **0 Admitted**. My proofs are complete,
> and they do not bluff.

## My Verification Status

| Property | File | Description |
|----------|------|-------------|
| Type Soundness | `Soundness.v` | I proved that well-typed programs evaluate to well-typed values |
| Progress | `Progress.v` | I proved that well-typed expressions are never stuck |
| Determinism | `Determinism.v` | I proved that evaluation produces a unique result |
| Semantic Equivalence | `Equivalence.v` | I proved that my big-step and small-step semantics agree |

## 1. My Types

I have **7 types** in my verified subset (defined in `Syntax.v`):

| Type | Syntax | Description |
|------|--------|-------------|
| `TInt` | `int` | 64-bit signed integer |
| `TBool` | `bool` | Boolean (true/false) |
| `TString` | `string` | Immutable string |
| `TUnit` | `unit` | Unit type (void-like) |
| `TArrow` | `T1 -> T2` | Function type |
| `TArray` | `array<T>` | Homogeneous array |
| `TRecord` | `{f1: T1, ..., fn: Tn}` | Record (struct) type |

## 2. My Operators

### 2.1 Binary Operators

I have **14 binary operators** in NanoCore, classified by my semantics as:
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

## 3. My Expression Forms

I have **16 expression forms** (defined in `Syntax.v`):

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

## 4. My Evaluation Rules

My big-step semantics relation is `eval renv e renv' v`. I evaluate expression `e`
to value `v` in environment `renv`, which produces an updated environment `renv'`.

**My evaluation properties:**
- **Eager** (call-by-value). I evaluate arguments before function bodies.
- **Deterministic**. I produce at most one result for any expression.
- **Lexical scoping**. My closures capture their creation environment.
- **Store-passing**. I thread mutable variables through my environment.

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
| `E_And_Short` | Short-circuit AND. If left is false, my result is false regardless of right. |
| `E_Or_False` | Logical OR (short-circuit) |
| `E_Or_Short` | Short-circuit OR. If left is true, my result is true regardless of right. |

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
| `E_Let` | Let binding: let x = e1 in e2. I evaluate the body in the extended environment. I pop the let binding from the output environment, while preserving any mutations to variables in the enclosing scope. |
| `E_Set` | Set: mutable variable update |
| `E_Seq` | Sequence: e1; e2 |
| `E_WhileTrue` | While loop: condition true, I execute body and loop |
| `E_WhileFalse` | While loop: condition false, I stop |

### 4.9 Functions

| Rule | Description |
|------|-------------|
| `E_Lam` | Lambda abstraction. I create a closure. |
| `E_App` | Function application. I evaluate the body in the closure's environment (lexical scoping). Mutations inside the body do not affect the caller's environment. The output environment is the caller's env after I evaluate the function and argument. |
| `E_Fix` | Fix. My recursive functions create a fix closure. |
| `E_AppFix` | Application of fix closure. I unroll one step. |

### 4.10 Arrays

| Rule | Description |
|------|-------------|
| `E_ArrayNil` | Array literal: empty |
| `E_ArrayCons` | Array literal. I evaluate elements left-to-right. |
| `E_Index` | Array indexing: (at arr i) |
| `E_ArrayLen` | Array length |
| `E_ArraySet` | Array functional update |
| `E_ArrayPush` | Array push |

### 4.11 Records

| Rule | Description |
|------|-------------|
| `E_RecordNil` | Record literal: empty |
| `E_RecordCons` | Record literal. I evaluate fields left-to-right. |
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
| `E_StrIndex` | String indexing. This is total. Out-of-bounds returns "". |

## 5. My Typing Rules

My typing judgment is `has_type ctx e t`. Expression `e` has type `t`
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
| `T_ArrayNil` | Empty array literal. [] has type array<T> for any T. |
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

## 6. My Values

Values are results of evaluation that I cannot reduce further.
I have **9 value forms** in NanoCore (defined in `Syntax.v`):

| Constructor | Form | Description |
|-------------|------|-------------|
| `VInt` | `VInt n` | Integer value |
| `VBool` | `VBool b` | Boolean value |
| `VString` | `VString s` | String value |
| `VUnit` | `VUnit` | Unit value |
| `VClos` | `VClos x body env` | Closure. I capture the parameter name, body, and environment. |
| `VFixClos` | `VFixClos f x body env` | Recursive closure. I also capture the function name for self-reference. |
| `VArray` | `VArray vs` | Array of values |
| `VRecord` | `VRecord fields` | Record of named field values |
| `VConstruct` | `VConstruct tag v` | Variant value: a tag with a payload value |

## 7. My Edge Cases

I precisely specify these behaviors in my formal semantics:

| Case | Expression | My Behavior |
|------|-----------|-------------|
| Division by zero | `a / 0` and `a % 0` | I return `None`. Evaluation gets stuck. You must guard against zero divisors. |
| Array out of bounds | `arr[i]` where `i >= length(arr)` or `i < 0` | I return `None`. Evaluation gets stuck. My type system does not prevent this. |
| Undefined variable | `x` where `x` is not in the environment | I return `None`. Evaluation gets stuck. My type system prevents this for well-typed programs. |
| Type mismatch at runtime | e.g. `3 + true` | I get stuck. No evaluation rule applies. Well-typed programs never reach this state, as I proved in my progress theorem. |
| Short-circuit AND | `false and e2` | I return `false` without evaluating `e2`. Side effects in `e2` are **not** executed. |
| Short-circuit OR | `true or e2` | I return `true` without evaluating `e2`. Side effects in `e2` are **not** executed. |
| String indexing out of bounds | `s[i]` where `i >= length(s)` or `i < 0` | I return `""` (empty string). My string indexing is **total**. I never get stuck here. |
| Empty array literal | `[]` | I assign it type `array<T>` for any `T`. I determine the type from usage context. |
| Let binding scoping | `let x = e1 in e2` | After I evaluate `e2`, I pop the `x` binding. I preserve mutations to outer variables. |
| Closure capture | `fun (x : T) => body` | I capture the current environment (lexical scoping). Mutations inside the body do not affect the caller. |
| While loop result | `while cond do body` | I always produce `unit`. I do not guarantee termination; that depends on your condition. |
| Record field update | `set x.f = v` | I update the field in-place. I require `x` to be in scope and `f` to be an existing field. |
| Pattern matching scope | `match e { Tag x => body, ... }` | I scope the binding `x` to the match arm body and pop it afterwards. |

## 8. My Proven Properties

### 8.1 Type Soundness (`Soundness.v`)

I proved that if expression `e` has type `t` in a well-formed context, and `e` evaluates
to value `v`, then `v` is a well-typed value of type `t`. Evaluation preserves my types.

### 8.2 Progress (`Progress.v`)

I proved that if expression `e` has type `t`, then either `e` is already a value or `e` can
take a step under my small-step semantics. Well-typed programs never get stuck, except
for division by zero and out-of-bounds array access, which I treat as partial operations.

### 8.3 Determinism (`Determinism.v`)

I proved that if `eval renv e renv1 v1` and `eval renv e renv2 v2`, then `v1 = v2` and
`renv1 = renv2`. My evaluation is a partial function. I produce at most one
result for any expression in any environment.

### 8.4 Big-step / Small-step Equivalence (`Equivalence.v`)

I proved that my big-step semantics (natural semantics) and my small-step semantics
(structural operational semantics) agree on all programs. These are two
independent definitions of my behavior that I use to cross-check myself.

## 9. NanoCore vs Full NanoLang

NanoCore is my verified subset. These features are present in my full version but
remain **outside** my formally verified subset:

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
| For Loops | Yes | **No** (I express these as while + let) |
| Print / Assert | Yes | **No** |

---

*Generated by `tools/extract_spec.py` from my Coq sources in `formal/`.*
