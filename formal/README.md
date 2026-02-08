# NanoCore: Formal Verification (Phase 4)

Mechanized metatheory for NanoCore, a minimal subset of NanoLang,
formalized in the Rocq Prover (Coq).

## What's proved

**Type soundness** via preservation + progress, plus **determinism**:

**Preservation:** If a well-typed expression evaluates to a value,
that value has the expected type, and environment agreement is
maintained through mutation. (Big-step semantics with store-passing.)

```
Theorem preservation : forall renv e renv' result,
  eval renv e renv' result ->
  forall gamma t,
  has_type gamma e t ->
  env_ctx_agree renv gamma ->
  val_has_type result t /\ env_ctx_agree renv' gamma.
```

**Progress:** A well-typed closed expression is either a value or can
take a reduction step. While loops unroll to if expressions, set
reduces to unit. (Small-step semantics, division by zero defined
to produce 0.)

```
Theorem progress : forall e t,
  has_type CtxNil e t ->
  is_value e \/ (exists e', step e e').
```

**Determinism:** The big-step evaluation relation is a partial function:
if an expression evaluates to two results, they are identical.

```
Theorem eval_deterministic : forall renv e renv' v renv'' v',
  eval renv e renv' v ->
  eval renv e renv'' v' ->
  v = v' /\ renv' = renv''.
```

## NanoCore subset

| Feature | Included |
|---------|----------|
| Integer and boolean literals | Yes |
| String literals | Yes |
| Unit type and literal | Yes |
| Arithmetic (`+`, `-`, `*`, `/`, `%`) | Yes |
| Comparison (`==`, `!=`, `<`, `<=`, `>`, `>=`) | Yes |
| Logical (`and`, `or`, `not`) | Yes |
| String concatenation (`+`) | Yes |
| String length (`str_length`) | Yes |
| String equality (`==`, `!=`) | Yes |
| If/then/else | Yes |
| Let bindings | Yes |
| Mutable variables (`set`) | Yes |
| Sequential composition (`;`) | Yes |
| While loops | Yes |
| Lambda / application | Yes |
| Array literals (`[e1, ..., en]`) | Yes |
| Array indexing (`at`) | Yes |
| Array length (`array_length`) | Yes |
| Record (struct) literals | Yes |
| Record field access (`.f`) | Yes |

## File structure

| File | Contents |
|------|----------|
| `Syntax.v` | Types, operators, expressions, values, environments, env_update |
| `Semantics.v` | Big-step operational semantics with store-passing |
| `Typing.v` | Typing rules and contexts |
| `Soundness.v` | Preservation theorem (value typing + env agreement) |
| `Progress.v` | Small-step semantics, substitution, progress theorem |
| `Determinism.v` | Determinism of evaluation (eval is a partial function) |

## Building

Requires the Rocq Prover (>= 9.0). Install via:

```
brew install rocq-prover   # macOS
opam install rocq-prover   # or via opam
```

Then:

```
cd formal/
make
```

## Design choices

- **Big-step semantics with store-passing** for the preservation proof.
  The evaluation relation `eval renv e renv' v` threads the environment
  through all subexpressions, capturing mutable variable updates.
- **Scoped let bindings**: E_Let pops the let binding from the output
  environment, ensuring environment agreement is preserved across let
  boundaries. Inner mutations to enclosing-scope variables propagate
  correctly through `env_update`.
- **Lexically scoped function application**: E_App uses the caller's
  environment as output (not the closure body's), preventing mutations
  inside function bodies from leaking to call sites.
- **String-based environments** (association lists) rather than de Bruijn
  indices, matching the implementation
- **Induction on evaluation derivation** for the preservation proof,
  which naturally handles the closure body case in function application
- **Short-circuit AND/OR** modeled with separate evaluation rules
- **Small-step semantics** for the progress proof, avoiding the need
  for logical relations to prove termination
- **Total division** (div-by-zero produces 0) in small-step, making
  progress unconditional
- **While loop unrolling** in small-step: `while c b` steps to
  `if c then (b; while c b) else unit`
- **Set as unit producer** in small-step: `set x v` steps to `unit`
  (store update is modeled in big-step; small-step progress only
  requires showing one step exists)
- **Immutable array values** modeled as `list val` / `list expr`.
  Left-to-right element evaluation in small-step via `S_ArrayHead`
  / `S_ArrayTail` rules. Out-of-bounds indexing defaults to `unit`
  in small-step (total), while big-step uses `nth_error` (partial).
- **Nested inductive handling**: `subst` uses a local fixpoint for
  the `EArray (list expr)` and `ERecord (list (string * expr))` cases
  to satisfy the guard checker
- **Structural record typing**: Records use structural typing
  (`TRecord : list (string * ty) -> ty`) rather than nominal typing,
  keeping the formalization simple without a separate struct environment
- **Forall2-based record agreement**: `VT_Record` uses `Forall2` to
  relate value fields to type fields, ensuring field names match and
  values have the correct types
- **Polymorphic assoc_lookup**: Used for both type-level and value-level
  field lookup in records, with `ty_eq_dec` extended via `fix` to handle
  the nested `TRecord(list (string * ty))` case

## Phases

- **Phase 0:** Pure NanoCore (int, bool, unit, arithmetic, comparison, logic, if/let, lambda/app)
- **Phase 1:** Mutation and while loops (set, seq, while, store-passing semantics)
- **Phase 2:** Strings (string literals, concatenation, length, equality)
- **Phase 3:** Arrays (array literals, indexing, length)
- **Phase 4:** Records/structs (record literals, field access) -- current
