# Dependent Types & Contracts for NanoLang

**Status:** Design Phase (Incremental Approach)  
**Priority:** P1  
**Bead:** nanolang-o6fn  
**Created:** 2025-12-29

## Vision

Enable compile-time verification of program properties through dependent types, starting with simple contracts and evolving toward full dependent types.

## Problem: Runtime Errors That Could Be Caught at Compile Time

```nano
fn array_get<T>(arr: array<T>, idx: int) -> T {
    /* Runtime check - crashes if idx out of bounds */
    if (or (< idx 0) (>= idx (array_length arr))) {
        (panic "Index out of bounds")
    }
    unsafe {
        return (unsafe_array_get arr idx)
    }
}
```

**Why not catch at compile time?**

With dependent types:
```nano
fn array_get<T>(arr: array<T>, idx: int) -> T
    requires (and (>= idx 0) (< idx (array_length arr)))
{
    /* No runtime check needed! */
    unsafe {
        return (unsafe_array_get arr idx)
    }
}
```

## Incremental Approach

Full dependent types are **extremely complex**. We take a phased approach:

### Phase 1: Simple Contracts (6 months)
- Preconditions and postconditions
- Runtime checks by default
- Static analysis for obvious cases

### Phase 2: Refinement Types (12 months)
- Types with predicates
- SMT solver integration
- Bounded integers, non-empty lists

### Phase 3: Full Dependent Types (24+ months)
- Types depending on values
- Proof obligations
- Curry-Howard correspondence

## Phase 1: Contracts (RECOMMENDED START)

### Syntax

```nano
fn divide(a: int, b: int) -> int
    requires (!= b 0)                    /* Precondition */
    ensures (== a (+ (* result b) (% a b)))  /* Postcondition */
{
    return (/ a b)
}

fn array_head<T>(arr: array<T>) -> T
    requires (> (array_length arr) 0)
{
    return (array_get arr 0)
}

fn sorted_insert(list: List<int>, value: int) -> List<int>
    requires (is_sorted list)
    ensures (is_sorted result)
    ensures (== (list_length result) (+ (list_length list) 1))
{
    /* Implementation */
}
```

### Semantics

**Precondition:**
- Checked at **call site**
- Caller must prove precondition holds
- Compiler warns if cannot prove

**Postcondition:**
- Checked at **return site**
- Callee must prove postcondition holds
- Uses `result` to refer to return value

### Checking Strategy

**Static Analysis (when possible):**
```nano
let x: int = 10
let y: int = 2
let z: int = (divide x y)  /* ✓ Compiler knows y != 0 */
```

**Runtime Check (when uncertain):**
```nano
fn process(a: int, b: int) -> int {
    /* Compiler cannot prove b != 0, inserts runtime check */
    return (divide a b)  /* Runtime check: assert (b != 0) */
}
```

**Explicit Proof (advanced):**
```nano
fn safe_divide(a: int, b: int) -> int {
    if (== b 0) {
        return 0
    } else {
        /* Compiler knows b != 0 in this branch */
        return (divide a b)  /* No check needed */
    }
}
```

### Implementation

**1.1 Parser Extensions**
```c
/* src/parser.c */
typedef struct {
    char *name;
    Type ret_type;
    Parameter *params;
    int param_count;
    Expr *preconditions;   /* NEW: array of requires clauses */
    int precond_count;
    Expr *postconditions;  /* NEW: array of ensures clauses */
    int postcond_count;
    Statement *body;
} Function;
```

**1.2 Typechecker Integration**
```c
/* src/typechecker.c */
void check_function_call(Function *fn, Expr **args, int arg_count) {
    /* Check preconditions */
    for (int i = 0; i < fn->precond_count; i++) {
        Expr *precond = fn->preconditions[i];
        
        /* Try to prove precondition statically */
        if (can_prove_statically(precond, args)) {
            /* OK, no runtime check needed */
        } else if (is_obviously_false(precond, args)) {
            /* Error: precondition will always fail */
            error("Precondition will never hold");
        } else {
            /* Warning: runtime check inserted */
            warning("Cannot verify precondition statically, inserting runtime check");
            insert_runtime_check(precond, args);
        }
    }
}
```

**1.3 Transpiler: Runtime Checks**
```c
/* src/transpiler.c */
void transpile_function(Function *fn) {
    /* Generate precondition checks */
    for (int i = 0; i < fn->precond_count; i++) {
        emit("if (!(%s)) {", transpile_expr(fn->preconditions[i]));
        emit("    fprintf(stderr, \"Precondition failed: %s\\n\", \"%s\");",
             expr_to_string(fn->preconditions[i]),
             expr_to_string(fn->preconditions[i]));
        emit("    abort();");
        emit("}");
    }
    
    /* Generate function body */
    transpile_stmt(fn->body);
    
    /* Generate postcondition checks */
    for (int i = 0; i < fn->postcond_count; i++) {
        emit("if (!(%s)) {", transpile_postcondition(fn->postconditions[i]));
        emit("    fprintf(stderr, \"Postcondition failed: %s\\n\", \"%s\");",
             expr_to_string(fn->postconditions[i]),
             expr_to_string(fn->postconditions[i]));
        emit("    abort();");
        emit("}");
    }
}
```

### Timeline: 6 months
- Month 1-2: Parser + AST extensions
- Month 3-4: Typechecker integration
- Month 5: Transpiler (runtime checks)
- Month 6: Testing + documentation

## Phase 2: Refinement Types (Future)

### Syntax

```nano
/* Refined integer types */
type Positive = int { x > 0 }
type NonZero = int { x != 0 }
type Port = int { x >= 0 and x < 65536 }
type Index<n> = int { x >= 0 and x < n }

/* Refined array types */
type NonEmptyArray<T> = array<T> { (array_length arr) > 0 }

/* Function using refinement types */
fn divide(a: int, b: NonZero) -> int {
    /* No precondition needed! Type system ensures b != 0 */
    return (/ a b)
}

fn array_head<T>(arr: NonEmptyArray<T>) -> T {
    /* No precondition needed! Type system ensures arr non-empty */
    return (array_get arr 0)
}
```

### SMT Solver Integration

Use Z3 to prove predicates:
```nano
fn example(x: Positive) -> Positive {
    let y: int = (+ x 1)
    /* Z3 proves: x > 0 ⟹ x + 1 > 0 */
    return y  /* Type checks! */
}
```

### Implementation Challenges

- **Subtyping:** Positive <: int
- **Type inference:** Inferring refined types
- **SMT integration:** Z3 bindings, query generation
- **Performance:** SMT queries can be slow

### Timeline: 12 months
- Month 1-3: Type system extensions
- Month 4-6: Z3 integration
- Month 7-9: Type inference
- Month 10-12: Testing + optimization

## Phase 3: Full Dependent Types (Far Future)

### Syntax

```nano
/* Vec: array indexed by length */
type Vec<T, n: int> = array<T> { (array_length arr) == n }

/* Safe array operations */
fn vec_get<T, n: int>(v: Vec<T, n>, i: int) -> T
    requires (i < n)
{
    return (array_get v i)
}

/* Matrix with dimensions in type */
type Matrix<m: int, n: int> = Vec<Vec<float, n>, m>

/* Matrix multiplication with dimension checking */
fn matmul<m: int, n: int, p: int>(
    a: Matrix<m, n>,
    b: Matrix<n, p>
) -> Matrix<m, p> {
    /* Type system ensures dimensions match! */
    /* ... */
}

/* Printf-style format checking */
fn printf<T1, T2>(fmt: FormatString<T1, T2>, arg1: T1, arg2: T2) -> string {
    /* Type system checks format string matches argument types */
    /* ... */
}
```

### Proof Obligations

```nano
fn factorial(n: int) -> int
    requires (>= n 0)
    ensures (>= result 1)
{
    if (== n 0) {
        return 1  /* Proves: 1 >= 1 */
    } else {
        let rec: int = (factorial (- n 1))
        /* Need to prove: rec >= 1 ⟹ n * rec >= 1 */
        return (* n rec)
    }
}
```

### Implementation Challenges

- **Kind system:** Types have kinds (* -> *), (* -> * -> *)
- **Type equality:** Definitional vs propositional
- **Proof search:** Automated theorem proving
- **Decidability:** Typechecking may not terminate
- **Dual implementation:** C + NanoLang = 2× complexity!

### Timeline: 24+ months (PhD-level work)

## Recommendation: Start with Phase 1

**Phase 1 (Contracts)** is:
- ✅ Achievable in 6 months
- ✅ Provides immediate value
- ✅ Foundation for future phases
- ✅ Manageable dual implementation

**Phase 2 (Refinements)** is:
- ⚠️ Significant complexity
- ⚠️ Requires SMT solver integration
- ⚠️ 12 month effort
- ✅ Publishable research

**Phase 3 (Full Dependent Types)** is:
- ❌ Multi-year research project
- ❌ Requires PL theory expertise
- ❌ Dual implementation cost is VERY high
- ⚠️ May not align with NanoLang goals

## Alternative: Lightweight Verification

Instead of full dependent types, consider:

### Option A: Dafny-Style Contracts
- Contracts with automatic verification
- SMT-based checking
- No dependent types needed

### Option B: Eiffel-Style Design by Contract
- Runtime checks only
- No static verification
- Simpler, faster to implement

### Option C: Rust-Style Type States
- Encode states in types
- No predicates needed
- Linear types (see LINEAR_TYPES_DESIGN.md)

## Integration with Linear Types

Contracts + linear types = powerful combination:

```nano
type FileHandle = resource int

fn read(file: consume FileHandle) -> (string, FileHandle)
    requires (file_is_open file)
    ensures (file_is_open (get_second result))
{
    /* Contract ensures file remains valid */
    /* Linear types ensure single use */
    /* ... */
}
```

## Success Criteria

### Phase 1 (Contracts)
1. ✅ Can write requires/ensures clauses
2. ✅ Runtime checks generated by default
3. ✅ Static analysis for simple cases
4. ✅ Clear error messages
5. ✅ Documentation with examples
6. ✅ Stdlib uses contracts

### Phase 2 (Refinements)
1. ✅ Can define refined types
2. ✅ Z3 integration working
3. ✅ Subtyping implemented
4. ✅ Type inference for refinements
5. ✅ Performance acceptable (<1s per file)

### Phase 3 (Full Dependent Types)
1. ✅ Types can depend on values
2. ✅ Proof obligations generated
3. ✅ Automated proof search
4. ✅ Matrix example works
5. ✅ Published in top-tier venue (POPL/ICFP)

## Examples

### Example 1: Safe Division

```nano
fn divide(a: int, b: int) -> int
    requires (!= b 0)
{
    return (/ a b)
}

fn test_divide() -> void {
    let x: int = 10
    let y: int = 2
    let z: int = (divide x y)  /* ✓ OK: 2 != 0 */
    
    let w: int = 0
    let bad: int = (divide x w)  /* ✗ ERROR: Precondition may not hold */
}
```

### Example 2: Non-Empty List

```nano
fn list_head<T>(list: List<T>) -> T
    requires (> (list_length list) 0)
{
    return (list_get list 0)
}

fn process_list(items: List<int>) -> int {
    if (== (list_length items) 0) {
        return 0
    } else {
        /* Compiler knows list non-empty here */
        let first: int = (list_head items)  /* ✓ OK */
        return first
    }
}
```

### Example 3: Sorted List Invariant

```nano
fn is_sorted(list: List<int>) -> bool {
    let mut i: int = 0
    while (< i (- (list_length list) 1)) {
        let a: int = (list_get list i)
        let b: int = (list_get list (+ i 1))
        if (> a b) {
            return false
        }
        set i (+ i 1)
    }
    return true
}

fn sorted_insert(list: List<int>, value: int) -> List<int>
    requires (is_sorted list)
    ensures (is_sorted result)
    ensures (== (list_length result) (+ (list_length list) 1))
{
    /* Binary search + insert */
    /* Compiler checks postconditions */
    /* ... */
}
```

## References

### Dependent Types
- **Idris**: https://www.idris-lang.org/
- **Agda**: https://wiki.portal.chalmers.se/agda/
- **Coq**: https://coq.inria.fr/

### Refinement Types
- **Liquid Haskell**: https://ucsd-progsys.github.io/liquidhaskell/
- **F***: https://www.fstar-lang.org/
- **Dafny**: https://dafny.org/

### Contracts
- **Eiffel**: Design by Contract
- **Racket**: Contract system
- **Ada**: Preconditions/postconditions

### Papers
- "Refinement Types for ML" (Freeman & Pfenning, 1991)
- "Liquid Types" (Rondon et al., 2008)
- "Dependent Types in Practical Programming" (Xi & Pfenning, 1999)

---

**Recommendation:** Start with Phase 1 (Contracts) in Year 2  
**Timeline:** 6 months for MVP  
**Defer:** Phases 2-3 until Phase 1 proven valuable

