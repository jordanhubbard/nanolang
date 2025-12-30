# Formal Verification Roadmap for NanoLang

**Status:** Research & Long-Term Planning  
**Priority:** P1 (Research)  
**Bead:** nanolang-cmmk  
**Created:** 2025-12-29

## Vision

Mathematically prove that the NanoLang compiler produces correct code - that compiled programs behave exactly as specified by source semantics.

**Inspiration:** CompCert (verified C compiler), CakeML (verified ML), Vellvm (verified LLVM)

## Why Formal Verification?

### Current State
- **Bootstrapping** validates empirically that Stage1 == Stage2
- **Testing** provides confidence through examples
- **No proof** of correctness for arbitrary programs

### With Formal Verification
- **Mathematical certainty** that compiler is correct
- **Trust** for safety-critical applications (medical, automotive, aerospace)
- **Bug prevention** - proofs reveal subtle bugs
- **Publication-worthy** research contribution

## The Challenge

This is a **multi-year, multi-person research project**. CompCert took 6+ years with a team of experts.

**Realistic timeline:** 3-5 years for core verification  
**Required expertise:** Formal methods, proof assistants (Coq/Isabelle/Lean), PL theory

## Phased Approach

### Phase 0: Foundations (Year 1)

**Goal:** Establish formal semantics and proof infrastructure

#### 0.1 Choose Proof Assistant
- **Coq**: Most mature, CompCert uses it
- **Isabelle/HOL**: Strong automation
- **Lean 4**: Modern, good performance
- **Recommendation:** Start with Coq (most resources available)

#### 0.2 Formal Semantics of NanoLang

Define operational semantics in Coq:

```coq
(* Values *)
Inductive value : Type :=
  | VInt : Z -> value
  | VFloat : R -> value
  | VBool : bool -> value
  | VString : string -> value
  | VClosure : env -> list ident -> expr -> value.

(* Expressions *)
Inductive expr : Type :=
  | EVar : ident -> expr
  | ELit : value -> expr
  | EBinOp : binop -> expr -> expr -> expr
  | EApp : expr -> list expr -> expr
  | ELet : ident -> expr -> expr -> expr.

(* Small-step operational semantics *)
Inductive step : env -> expr -> env -> expr -> Prop :=
  | StepVar : forall env x v,
      lookup env x = Some v ->
      step env (EVar x) env (ELit v)
  | StepBinOp : forall env op v1 v2 v3,
      eval_binop op v1 v2 = Some v3 ->
      step env (EBinOp op (ELit v1) (ELit v2)) env (ELit v3)
  (* ... more rules ... *).

(* Big-step semantics *)
Inductive eval : env -> expr -> value -> Prop :=
  | EvalLit : forall env v,
      eval env (ELit v) v
  | EvalVar : forall env x v,
      lookup env x = Some v ->
      eval env (EVar x) v
  (* ... more rules ... *).
```

#### 0.3 Formal Semantics of C Subset

Define the subset of C that NanoLang compiles to:

```coq
(* Reuse CompCert's Clight semantics *)
Require Import compcert.cfrontend.Clight.

(* Or define minimal subset *)
Inductive c_expr : Type :=
  | CVar : ident -> c_expr
  | CInt : Z -> c_expr
  | CBinOp : c_binop -> c_expr -> c_expr -> c_expr
  | CCall : ident -> list c_expr -> c_expr.

Inductive c_stmt : Type :=
  | CSkip : c_stmt
  | CAssign : ident -> c_expr -> c_stmt
  | CSeq : c_stmt -> c_stmt -> c_stmt
  | CIf : c_expr -> c_stmt -> c_stmt -> c_stmt
  | CWhile : c_expr -> c_stmt -> c_stmt
  | CReturn : c_expr -> c_stmt.
```

#### 0.4 Compiler Specification

State the main theorem:

```coq
Theorem compiler_correctness :
  forall (src : nano_program) (tgt : c_program),
    compile src = Some tgt ->
    forall (input : value) (output : value),
      nano_eval src input output ->
      c_eval tgt input output.
```

**Deliverables:**
- Formal semantics in Coq
- Compiler specification theorem
- Basic infrastructure (tactics, lemmas)

**Timeline:** 12 months  
**Effort:** 1-2 FTE

### Phase 1: Type System Soundness (Year 2)

**Goal:** Prove type safety (progress + preservation)

#### 1.1 Formalize Type System

```coq
Inductive has_type : env -> expr -> type -> Prop :=
  | TInt : forall env n,
      has_type env (ELit (VInt n)) TInt
  | TVar : forall env x t,
      lookup_type env x = Some t ->
      has_type env (EVar x) t
  | TBinOp : forall env op e1 e2 t1 t2 t3,
      has_type env e1 t1 ->
      has_type env e2 t2 ->
      binop_type op t1 t2 = Some t3 ->
      has_type env (EBinOp op e1 e2) t3
  (* ... *).
```

#### 1.2 Prove Type Soundness

**Progress:** Well-typed terms don't get stuck
```coq
Theorem progress :
  forall e t,
    has_type empty e t ->
    is_value e \/ exists e', step empty e empty e'.
```

**Preservation:** Types are preserved by evaluation
```coq
Theorem preservation :
  forall env e e' t,
    has_type env e t ->
    step env e env e' ->
    has_type env e' t.
```

**Deliverables:**
- Formalized typechecker
- Progress + preservation theorems
- Soundness proof (no well-typed program crashes)

**Timeline:** 12 months  
**Effort:** 1-2 FTE

### Phase 2: Verify Core Passes (Year 3-4)

**Goal:** Verify individual compiler passes

#### 2.1 Lexer Verification

**Property:** Tokenization is invertible
```coq
Theorem lexer_inverse :
  forall (src : string) (tokens : list token),
    lex src = Some tokens ->
    unlex tokens = src.
```

**Complexity:** Low (lexers are simple)  
**Timeline:** 2 months

#### 2.2 Parser Verification

**Property:** AST correctly represents syntax
```coq
Theorem parser_correctness :
  forall (tokens : list token) (ast : expr),
    parse tokens = Some ast ->
    forall input output,
      eval_tokens tokens input output <->
      eval_ast ast input output.
```

**Complexity:** Medium (context-free grammar proof)  
**Timeline:** 6 months

#### 2.3 Typechecker Verification

**Property:** Typechecker implements formal type system
```coq
Theorem typechecker_correctness :
  forall env ast t,
    typecheck env ast = Some t <->
    has_type env ast t.
```

**Complexity:** Medium (already done in Phase 1)  
**Timeline:** 4 months (extraction + proof)

#### 2.4 Transpiler Verification ⭐ **HARDEST PART**

**Property:** Semantic preservation
```coq
Theorem transpiler_correctness :
  forall (nano_ast : nano_expr) (c_code : c_stmt),
    transpile nano_ast = Some c_code ->
    forall input output,
      nano_eval nano_ast input output ->
      c_eval c_code input output.
```

**Complexity:** Very High (main theorem)  
**Timeline:** 18 months  
**Challenges:**
- GC semantics (C has manual memory management)
- Runtime library verification
- String/array operations
- Module system

**Deliverables:**
- Verified lexer
- Verified parser
- Verified typechecker
- Verified transpiler (partial - core subset)

**Timeline:** 18-24 months  
**Effort:** 2-3 FTE

### Phase 3: Runtime Library Verification (Year 4-5)

**Goal:** Verify runtime functions (GC, arrays, strings)

#### 3.1 GC Verification

Prove GC preserves program semantics:
```coq
Theorem gc_preserves_semantics :
  forall heap heap' env e,
    gc heap = heap' ->
    reachable_from env e heap = reachable_from env e heap' /\
    eval heap env e = eval heap' env e.
```

**Complexity:** Very High  
**Timeline:** 12 months

#### 3.2 Array Operations

Verify bounds checking and operations:
```coq
Theorem array_get_safe :
  forall arr idx,
    0 <= idx < array_length arr ->
    exists v, array_get arr idx = Some v.
```

**Timeline:** 6 months

#### 3.3 String Operations

Verify concatenation, slicing, etc.:
```coq
Theorem string_concat_assoc :
  forall s1 s2 s3,
    string_concat s1 (string_concat s2 s3) =
    string_concat (string_concat s1 s2) s3.
```

**Timeline:** 4 months

**Deliverables:**
- Verified GC
- Verified array library
- Verified string library

**Timeline:** 18-24 months  
**Effort:** 2-3 FTE

### Phase 4: End-to-End Verification (Year 5)

**Goal:** Connect all pieces into full compiler correctness

```coq
Theorem compiler_correctness_full :
  forall (source : string) (binary : c_program),
    compile_full source = Some binary ->
    forall input output,
      interp_source source input output ->
      exec_binary binary input output.
```

**Complexity:** Extremely High (integration)  
**Timeline:** 12 months  
**Effort:** 2-3 FTE

## Alternative: Partial Verification

Full verification is a **massive undertaking**. Consider partial approaches:

### Option A: Verify Core Subset

- Focus on pure functional subset (no I/O, no FFI)
- Skip GC verification (assume correct)
- Verify typechecker + transpiler only

**Timeline:** 2-3 years  
**Benefit:** Still publishable, proves type safety

### Option B: Translation Validation

Instead of verifying compiler, verify **each compilation**:
- Compile program normally
- Generate proof that THIS compilation is correct
- Faster feedback, no need for proof assistant mastery

**Timeline:** 1-2 years  
**Benefit:** Practical, catches bugs in real compiles

### Option C: Lightweight Formal Methods

Use lighter tools:
- **Z3/SMT solvers** for bounded verification
- **CBMC** for C code verification
- **Frama-C** for C annotation + proof

**Timeline:** 6-12 months  
**Benefit:** Catches real bugs without full formal proof

## Recommended Strategy

Given NanoLang's goals and resources:

### Near Term (Year 1)
1. **Phase 0:** Define formal semantics
2. **Lightweight methods:** Add Z3-based bounded checking
3. **Translation validation:** Prototype for simple programs

### Medium Term (Year 2-3)
1. **Phase 1:** Prove type soundness
2. **Partial verification:** Verify typechecker
3. **Publication:** Write paper on results

### Long Term (Year 4-5)
1. **Seek funding:** NSF grant, industry partnership
2. **Hire experts:** PhD students or postdocs
3. **Phase 2-4:** Full verification (if resources allow)

## Resources Required

### Expertise
- Formal methods expert (PhD-level)
- PL theory background
- Coq/Isabelle experience
- Compiler internals knowledge

### Tooling
- Coq proof assistant
- CompCert as reference
- CI for proof checking

### Time
- **Minimum:** 2-3 person-years for core verification
- **Full verification:** 5-10 person-years

## Success Metrics

### Phase 0 (Foundations)
- ✅ Formal semantics written
- ✅ Compiler spec stated
- ✅ Basic proofs working

### Phase 1 (Type Safety)
- ✅ Progress theorem proven
- ✅ Preservation theorem proven
- ✅ Type soundness paper published

### Phase 2-4 (Full Verification)
- ✅ Core passes verified
- ✅ Runtime verified
- ✅ End-to-end correctness theorem
- ✅ CompCert-level guarantee

## Publications

Verification work is highly publishable:
- **POPL**: Type soundness + partial verification
- **PLDI**: Full compiler verification
- **ICFP**: Functional subset verification
- **CPP**: Proof engineering techniques

## References

### Verified Compilers
- **CompCert**: https://compcert.org/
- **CakeML**: https://cakeml.org/
- **Vellvm**: https://www.cis.upenn.edu/~stevez/vellvm/
- **Pilsner**: Verified PHP compiler

### Proof Assistants
- **Coq**: https://coq.inria.fr/
- **Isabelle/HOL**: https://isabelle.in.tum.de/
- **Lean 4**: https://lean-lang.org/

### Books
- **Software Foundations** (Pierce et al.)
- **Certified Programming with Dependent Types** (Chlipala)
- **Programs and Proofs** (Sergey)

---

**Status:** Roadmap document  
**Decision:** Defer full verification to future (research project)  
**Near-term action:** Focus on Phase 0 (formal semantics) as documentation

