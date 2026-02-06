# NanoCore: Formal Verification (Phase 0)

Mechanized metatheory for NanoCore, a minimal subset of NanoLang,
formalized in the Rocq Prover (Coq).

## What's proved

**Type soundness** via preservation + progress:

**Preservation:** If a well-typed expression evaluates to a value,
that value has the expected type. (Big-step semantics.)

```
Theorem preservation : forall renv e result,
  eval renv e result ->
  forall gamma t,
  has_type gamma e t ->
  env_ctx_agree renv gamma ->
  val_has_type result t.
```

**Progress:** A well-typed closed expression is either a value or can
take a reduction step. (Small-step semantics, division by zero defined
to produce 0.)

```
Theorem progress : forall e t,
  has_type CtxNil e t ->
  is_value e \/ (exists e', step e e').
```

## NanoCore subset

| Feature | Included |
|---------|----------|
| Integer and boolean literals | Yes |
| Arithmetic (`+`, `-`, `*`, `/`, `%`) | Yes |
| Comparison (`==`, `!=`, `<`, `<=`, `>`, `>=`) | Yes |
| Logical (`and`, `or`, `not`) | Yes |
| If/then/else | Yes |
| Let bindings | Yes |
| Lambda / application | Yes |
| Mutation, loops, strings | No (Phase 1+) |

## File structure

| File | Contents |
|------|----------|
| `Syntax.v` | Types, operators, expressions, values, environments |
| `Semantics.v` | Big-step operational semantics |
| `Typing.v` | Typing rules and contexts |
| `Soundness.v` | Preservation theorem and supporting lemmas |
| `Progress.v` | Small-step semantics, substitution, progress theorem |

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

- **Big-step semantics** for simplicity and direct correspondence with
  NanoLang's tree-walking interpreter
- **String-based environments** (association lists) rather than de Bruijn
  indices, matching the implementation
- **Induction on evaluation derivation** for the preservation proof,
  which naturally handles the closure body case in function application
- **Short-circuit AND/OR** modeled with separate evaluation rules
- **Small-step semantics** for the progress proof, avoiding the need
  for logical relations to prove termination
- **Total division** (div-by-zero produces 0) in small-step, making
  progress unconditional

## Future phases

- **Phase 1:** Mutation (store-passing), while loops (clock/fuel-based
  functional big-step, CakeML-style)
- **Phase 2:** Structs, unions, arrays
