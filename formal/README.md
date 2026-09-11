# NanoCore: Formal Verification (Phase 5)

I formalize NanoCore, a minimal subset of my language, in the Rocq Prover
(Coq). I distinguish proof source from successfully compiled and independently
checked theorem terms. Absence of `Axiom` or `Admitted` declarations alone
does not establish either a successful build or assumption-free theorems.
I provide theorem statements for preservation, progress, determinism, and
semantic equivalence. Evaluator soundness is partial: `EvalFn.v` proves
selected cases, several conditional on recursive soundness. It does not yet
contain the general `eval_fn_sound` theorem. Absence of `Admitted` tokens
does not establish that every advertised result has been implemented.

## What's proved

Current build status (2026-09-11): my fresh pinned Rocq 9.0.1 build passes
for all ten proof/test modules and `Assumptions.v`. All 28 named assumption
reports print `Closed under the global context`, and `rocqchk` independently
checks all eleven compiled libraries and their dependencies successfully.
This checks the existing theorems, not the missing general evaluator theorem.
Reproduce from the repository root with:

```bash
bash scripts/check_proofs_container.sh
```

I pin the container by digest, mount sources read-only, force recompilation
in a temporary copy, and run `make check` with `rocq compile` and `rocqchk`.
I invoke the checker directly because this image's `rocq check` launcher fails
to execute it even though it is installed on `PATH`.
Once compilation succeeds, `Assumptions.v` prints dependencies of the named
theorems and the checker rechecks the compiled libraries. Printed assumptions
still require review; this target does not certify compiler correspondence.

**Type soundness** via preservation + progress, **determinism**,
and **semantic equivalence** between big-step and small-step semantics:

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

**Semantic Equivalence:** Big-step evaluation and small-step reduction
agree for the pure (mutation-free) fragment. Type annotations on lambdas,
fix-points, and constructors are treated as computationally irrelevant.

```
Theorem eval_to_multistep : forall e v,
  pure e -> eclosed e ->
  eval ENil e ENil v ->
  multi_step_equiv e (val_to_expr v).
```

The generalized version handles environments and proves closure
well-formedness (`val_good`) throughout:

```
Theorem eval_to_multistep_gen : forall renv e renv' v,
  pure e -> eval renv e renv' v ->
  env_good renv -> all_vals_closed renv ->
  eclosed (close renv e) ->
  multi_step_equiv (close renv e) (val_to_expr v) /\ val_good v.
```

**Computable Evaluator:** I implement a fuel-based interpreter extractable
to OCaml. The following is a remaining proof obligation, not an existing
theorem:

```
Theorem eval_fn_sound : forall fuel renv e renv' v,
  eval_fn fuel renv e = Some (renv', v) ->
  eval renv e renv' v.
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
| Record field update (`set x.f = e`) | Yes |
| Recursive functions (`fix`/`letrec`) | Yes |
| Variant types / sum types | Yes |
| Pattern matching (`match`) | Yes |
| Array functional update | Yes |
| Array push (append) | Yes |
| String indexing | Yes |
| Heterogeneous tuples and static indexing | Yes |

## File structure

| File | Contents |
|------|----------|
| `Syntax.v` | Types, operators, expressions, values, environments and lookup/update helpers |
| `Semantics.v` | Big-step operational semantics with store-passing |
| `Typing.v` | Typing rules, contexts, mutual inductive `has_type`/`branches_type` |
| `Soundness.v` | Preservation theorem (value typing + env agreement) |
| `Progress.v` | Small-step semantics, substitution, progress theorem |
| `Determinism.v` | Determinism of evaluation (eval is a partial function) |
| `Equivalence.v` | Simulation of pure big-step evaluation by small-step reduction, modulo type annotations |
| `EvalFn.v` | Computable fuel-based evaluator with partial soundness lemmas |
| `EvalFnTests.v` | Reducible regression examples for reference-evaluator behavior |
| `Exhaustiveness.v` | Pattern coverage properties |
| `Assumptions.v` | Dependency reports for named theorems |
| `Extract.v` | OCaml extraction configuration for reference interpreter |

## Building

Requires the Rocq Prover (>= 9.0). Install via:

```
brew install rocq-prover   # macOS
opam install rocq-prover   # or via opam
```

Then:

```
cd formal/
make check COQC="rocq compile" COQCHK="rocqchk -silent"
make extract COQC="rocq compile"  # Extract OCaml reference interpreter
make nanocore-ref COQC="rocq compile"  # Build reference interpreter binary
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
- **Recursive closures via VFixClos**: `fix f (x:T1):T2 = body` creates
  a `VFixClos f x body env` closure. Application unrolls by binding
  both the argument and the recursive reference in the closure's env
- **Variant types with exhaustive matching**: `TVariant` carries an
  ordered list of constructor tags with payload types. `branches_type`
  (mutual inductive with `has_type`) enforces exhaustive, ordered
  branch coverage
- **Interleaved IH tactic for determinism**: `det_step_full` applies
  one IH then immediately resolves non-eval premises (like `find_branch`).
  This handles rules where variables introduced by non-eval premises
  appear in subsequent eval premises (E_Match pattern)
- **Expression equivalence modulo type annotations**: `expr_equiv` relates
  expressions that differ only in type annotations on lambdas, fix-points,
  and constructors (computationally irrelevant)
- **Closure environment substitution**: `close renv e` substitutes all
  environment bindings into an expression; `close_except x renv e`
  preserves variable `x` unsubstituted (for lambda bodies)
- **val_good predicate**: Ensures closure environments are well-formed
  (`env_good` + `all_vals_closed`), with VG_Clos/VG_FixClos strengthened
  with `eclosed` hypothesis
- **subst_preserves_subst_inv**: Key commutativity lemma—if
  `subst y t e = e`, then `subst y t (subst x s e) = subst x s e`
- **Fuel-based computable evaluator**: `eval_fn` uses standard decreasing
  fuel technique (as in CompCert/CertiCoq) with `Some/None` return type;
  selected soundness cases are proved; the general induction remains unfinished

## Phases

- **Phase 0:** Pure NanoCore (int, bool, unit, arithmetic, comparison, logic, if/let, lambda/app)
- **Phase 1:** Mutation and while loops (set, seq, while, store-passing semantics)
- **Phase 2:** Strings (string literals, concatenation, length, equality)
- **Phase 3:** Arrays (array literals, indexing, length)
- **Phase 4:** Records/structs (record literals, field access)
- **Phase 5:** Recursive functions (fix), variants + pattern matching, mutable record fields, array update/push, string indexing, semantic equivalence, computable evaluator -- current

## Evidence boundary

I use compilation, named theorem assumptions, and independent library checking
as proof evidence. I do not use source line counts as a correctness metric.
General evaluator soundness and correspondence with my production compiler
and VM remain unfinished.
My reference evaluator short-circuits `and`/`or`, matching my big-step rules.
`eval_fn_and_short` and `eval_fn_or_short` state skipped-right-operand behavior
for arbitrary expressions and preserve the left evaluation's environment.
`eval_fn_sound_logic` proves logical-operator soundness conditional on sound
recursive evaluations; it does not supply the general evaluator theorem.
`EvalFnTests.v` checks truth tables, skipped stuck expressions and assignments,
necessary right-side effects, preserved left-side effects, and operand types.
It also checks zero divisors, let and match shadowing, outer mutation, closure
isolation, recursive calls, and loop state. My fourteen regression examples
complement the theorem statements.

I have conditional soundness lemmas for all binary operators, `let`, loops,
ordinary and recursive closure application, and variant matching, in addition
to the earlier cases. `eval_preserves_env_names` proves that my relational
evaluation preserves binding names and their order; this justifies removing
the bound slot after a let or match body. Aggregate cases and the final fuel
induction remain necessary before I can claim general evaluator soundness.
