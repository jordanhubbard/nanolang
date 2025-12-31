# Full Generics Implementation Roadmap for NanoLang

**Status**: Design Document  
**Created**: 2025-12-30  
**Complexity**: Very High (6-12 month effort)  

---

## Executive Summary

NanoLang currently supports **limited generics** via compile-time monomorphization for `List<T>`. Full generics would require:

1. **Generic function definitions**: `fn map<T, U>(f: fn(T) -> U, list: List<T>) -> List<U>`
2. **Generic structs**: `struct Pair<T, U> { first: T, second: U }`
3. **Generic enums/unions**: `enum Option<T> { Some(T), None }`
4. **Type inference**: Deduce type parameters from usage
5. **Trait/interface system**: Constrain type parameters (`T: Display`)
6. **Higher-kinded types** (optional): `F<T>` where `F` itself is a type parameter

**Current State:**
- ✅ Monomorphic `List<T>` via name mangling (`list_Point_new`)
- ✅ Environment tracks generic instantiations
- ✅ Transpiler generates C code for each `List<T>` specialization
- ❌ No generic functions (every function is monomorphic)
- ❌ No generic structs (only List is generic)
- ❌ No type inference for generics
- ❌ No trait/interface constraints

---

## Current Architecture Analysis

### What Works Today

```nano
struct Point { x: int, y: int }

fn process_points() -> int {
    let points: List<Point> = (list_Point_new)  // ✅ Works
    (list_Point_push points Point { x: 1, y: 2 })
    let p: Point = (list_Point_get points 0)
    return p.x
}
```

**Implementation:**
1. **Parser** recognizes `List<Point>` syntax
2. **Type checker** validates `Point` exists and tracks instantiation
3. **Transpiler** generates:
   ```c
   typedef struct {
       struct nl_Point *data;
       int count;
       int capacity;
   } List_Point;
   
   List_Point* list_Point_new(void);
   void list_Point_push(List_Point *list, struct nl_Point value);
   struct nl_Point list_Point_get(List_Point *list, int index);
   ```

4. **Runtime** includes generated `/tmp/list_Point.h` or inline definitions

### What Doesn't Work

```nano
// ❌ Generic functions
fn map<T, U>(f: fn(T) -> U, xs: List<T>) -> List<U> {
    let result: List<U> = (list_U_new)  // Don't know U at compile time
    for x in xs {
        (list_U_push result (f x))
    }
    return result
}

// ❌ Generic structs
struct Pair<T, U> {
    first: T,
    second: U
}

// ❌ Generic enums
enum Option<T> {
    Some(T),
    None
}

// ❌ Type inference
let x = (map inc numbers)  // Can't infer <int, int>
```

---

## Implementation Phases

### Phase 1: Generic Function Definitions (8-12 weeks)

**Goal**: Support generic functions with explicit type parameters.

```nano
fn identity<T>(x: T) -> T {
    return x
}

// Usage requires explicit instantiation
let a: int = (identity<int> 42)
let b: string = (identity<string> "hello")
```

**Required Changes:**

#### 1.1 Parser (`src/parser.c`)
```c
// Add AST node for type parameters
typedef struct {
    char **type_param_names;     // ["T", "U"]
    int type_param_count;
    ASTNode *body;
} ASTGenericFunction;

// Parse syntax: fn name<T, U>(args...) -> ReturnType { body }
ASTNode* parse_generic_function(Parser *p) {
    // 1. Parse 'fn'
    // 2. Parse function name
    // 3. Check for '<'
    // 4. Parse type parameters: T, U, V
    // 5. Parse '>'
    // 6. Parse regular parameters
    // 7. Parse return type
    // 8. Parse body
}
```

**Parser Complexity**: ~500 lines, moderate risk of breaking existing parsing

#### 1.2 Environment (`src/env.c`)
```c
// Store generic function templates
typedef struct {
    char *function_name;
    char **type_params;        // ["T", "U"]
    int type_param_count;
    ASTNode *body_template;    // Uninstantiated AST
    // ... parameter types, return type ...
} GenericFunctionTemplate;

// Track instantiations
typedef struct {
    char *template_name;       // "map"
    char **type_args;          // ["int", "string"]
    char *mangled_name;        // "map_int_string"
    ASTNode *instantiated_body;
} GenericFunctionInstance;

// Add to Environment
GenericFunctionTemplate *generic_templates;
int generic_template_count;
GenericFunctionInstance *generic_instances;
int generic_instance_count;
```

**Environment Complexity**: ~800 lines, high risk (core data structure)

#### 1.3 Type Checker (`src/typechecker.c`)
```c
// Type substitution for generic instantiation
Type substitute_type_params(Type t, TypeParamMap *map) {
    // T -> int, U -> string
    // List<T> -> List<int>
    // fn(T) -> U => fn(int) -> string
}

// Instantiate generic function
ASTNode* instantiate_generic_function(
    GenericFunctionTemplate *template,
    Type *type_args,
    int type_arg_count
) {
    // 1. Create type parameter map: T -> int, U -> string
    // 2. Clone AST body
    // 3. Walk AST, substituting type parameters
    // 4. Return instantiated AST
}

// Check generic function call
Type check_generic_function_call(
    const char *function_name,
    Type *type_args,
    ASTNode **call_args
) {
    // 1. Find generic template
    // 2. Instantiate with type_args
    // 3. Type check instantiated body
    // 4. Cache instantiation
    // 5. Return return type
}
```

**Type Checker Complexity**: ~1200 lines, **very high risk** (complex logic)

#### 1.4 Transpiler (`src/transpiler.c`)
```c
// Generate C code for each instantiation
void transpile_generic_instances(Environment *env, StringBuilder *sb) {
    for (int i = 0; i < env->generic_instance_count; i++) {
        GenericFunctionInstance *inst = &env->generic_instances[i];
        
        // Generate: int map_int_string(fn_ptr f, List_int xs) { ... }
        sb_appendf(sb, "%s %s(", 
                  get_c_type(inst->return_type),
                  inst->mangled_name);
        // ... parameters ...
        transpile_statement(inst->instantiated_body, sb);
    }
}
```

**Transpiler Complexity**: ~600 lines, moderate risk

**Phase 1 Total**: ~3100 lines of new/modified code, **12 weeks**

---

### Phase 2: Generic Structs (6-8 weeks)

**Goal**: Support user-defined generic structs.

```nano
struct Pair<T, U> {
    first: T,
    second: U
}

let p: Pair<int, string> = Pair { first: 42, second: "hello" }
```

**Required Changes:**

#### 2.1 Parser
```c
// Parse: struct Name<T, U> { fields... }
ASTNode* parse_generic_struct(Parser *p) {
    // Similar to generic functions
}
```

#### 2.2 Environment
```c
typedef struct {
    char *struct_name;
    char **type_params;
    FieldDef *fields;          // May reference type params
    int field_count;
} GenericStructTemplate;

typedef struct {
    char *template_name;       // "Pair"
    char **type_args;          // ["int", "string"]
    char *mangled_name;        // "Pair_int_string"
    FieldDef *concrete_fields; // Substituted fields
} GenericStructInstance;
```

#### 2.3 Type Checker
```c
// Instantiate struct when used
StructDef* instantiate_generic_struct(
    GenericStructTemplate *template,
    Type *type_args
) {
    // 1. Create type map
    // 2. Clone fields
    // 3. Substitute T, U in field types
    // 4. Register concrete struct
}
```

#### 2.4 Transpiler
```c
// Generate: typedef struct { int first; char* second; } Pair_int_string;
void transpile_generic_struct_instances(Environment *env, StringBuilder *sb) {
    for (int i = 0; i < env->generic_struct_instance_count; i++) {
        // Generate C struct definition
    }
}
```

**Phase 2 Total**: ~1800 lines, **8 weeks**

---

### Phase 3: Generic Enums/Unions (4-6 weeks)

**Goal**: Support generic unions.

```nano
enum Option<T> {
    Some(T),
    None
}

enum Result<T, E> {
    Ok(T),
    Err(E)
}
```

**Similar to Phase 2**, but for union types.

**Phase 3 Total**: ~1200 lines, **6 weeks**

---

### Phase 4: Type Inference (8-12 weeks)

**Goal**: Infer type parameters from usage.

```nano
// Before: explicit
let x: int = (identity<int> 42)

// After: inferred
let x: int = (identity 42)  // Infers T = int
```

**Required Changes:**

#### 4.1 Type Inference Engine
```c
// Unification algorithm (Hindley-Milner style)
typedef struct {
    Type expected;
    Type actual;
} TypeConstraint;

bool unify(Type t1, Type t2, TypeSubstitution *subst) {
    // 1. If t1 is type variable, bind to t2
    // 2. If t2 is type variable, bind to t1
    // 3. If both concrete, check equality
    // 4. If parametric, recurse on arguments
}

// Infer type arguments for generic call
Type* infer_type_arguments(
    GenericFunctionTemplate *template,
    ASTNode **call_args,
    int arg_count
) {
    // 1. Create fresh type variables for each type param
    // 2. Generate constraints from parameter types
    // 3. Run unification
    // 4. Solve for type variables
    // 5. Return concrete types or error
}
```

**Type Inference Complexity**: ~2000 lines, **very high risk** (complex algorithm)

**Phase 4 Total**: ~2500 lines, **12 weeks**

---

### Phase 5: Trait/Interface System (12-16 weeks)

**Goal**: Constrain generic type parameters.

```nano
trait Display {
    fn to_string(self) -> string
}

impl Display for Point {
    fn to_string(self) -> string {
        return (string_concat "Point(" (int_to_string self.x) ")")
    }
}

fn print_all<T: Display>(items: List<T>) -> void {
    for item in items {
        (println (item.to_string))
    }
}
```

**Required Changes:**

#### 5.1 Trait Definitions
```c
typedef struct {
    char *trait_name;
    FunctionSignature *required_methods;
    int method_count;
} TraitDef;
```

#### 5.2 Trait Implementations
```c
typedef struct {
    char *trait_name;
    char *type_name;
    ASTNode **method_impls;
    int method_count;
} TraitImpl;
```

#### 5.3 Constraint Checking
```c
bool satisfies_constraint(Type t, char *trait_name, Environment *env) {
    // 1. Find trait definition
    // 2. Find impl for type t
    // 3. Verify all methods implemented
}
```

#### 5.4 Trait Dispatch
```c
// Generate vtable for trait objects
typedef struct {
    void *data;
    void **vtable;  // Function pointers for trait methods
} TraitObject;
```

**Phase 5 Total**: ~4000 lines, **16 weeks**

---

## Total Effort Estimate

| Phase | Description | Lines of Code | Time | Risk |
|-------|-------------|---------------|------|------|
| 1 | Generic Functions | ~3100 | 12 weeks | High |
| 2 | Generic Structs | ~1800 | 8 weeks | Medium |
| 3 | Generic Enums/Unions | ~1200 | 6 weeks | Medium |
| 4 | Type Inference | ~2500 | 12 weeks | Very High |
| 5 | Traits/Interfaces | ~4000 | 16 weeks | Very High |
| **Total** | **Full Generics** | **~12,600** | **54 weeks** | **Very High** |

**Realistically**: 12-18 months for a single experienced compiler developer.

---

## Alternative Approach: Staged Implementation

### Minimum Viable Generics (MVG) - 3 months

Focus on **generic functions only**, no inference:

```nano
fn identity<T>(x: T) -> T { return x }

// Explicit instantiation required
(identity<int> 42)
(identity<string> "hello")
```

**Benefits:**
- ✅ Unlocks most practical use cases
- ✅ Simpler type checker (no inference)
- ✅ Builds on existing monomorphization
- ✅ Low risk to existing functionality

**Limitations:**
- ❌ Verbose syntax (explicit type args)
- ❌ No generic structs/enums
- ❌ No trait constraints

**Effort**: ~3000 lines, 12 weeks, high risk but manageable

---

## Comparison with Other Languages

### Rust Approach
- **Monomorphization** (like NanoLang's List)
- **Trait bounds** for constraints
- **Type inference** for ergonomics
- **Zero runtime cost**

**NanoLang could follow this model.**

### Swift/C# Approach
- **Reified generics** (runtime type information)
- **JIT compilation** or runtime specialization
- **Dynamic dispatch** for interfaces

**Not suitable for NanoLang** (systems language, no JIT)

### C++ Templates
- **Turing-complete** template metaprogramming
- **Duck typing** (implicit constraints)
- **Complex error messages**

**Avoid this** - too complex, poor errors

---

## Risks and Challenges

### 1. Type Checker Complexity
**Problem**: Generic type checking is fundamentally harder.
- Must track type variable bindings
- Must perform substitution correctly
- Must handle recursive types
- Must validate constraints

**Mitigation**: Start with explicit type args (no inference)

### 2. Error Messages
**Problem**: Generic errors are notoriously hard to understand.
```
Error: Cannot unify type 'T' with 'int' in context of generic function 'map'
  where T is bound to 'string' from argument 1
  but expected 'int' from return type constraint
```

**Mitigation**: Invest heavily in error message quality from day 1

### 3. Compilation Time
**Problem**: Monomorphization explodes compile times.
- `map<int, int>`, `map<string, string>`, `map<Point, Player>` all generate separate code
- Combinatorial explosion with nested generics

**Mitigation**: Incremental compilation, caching instantiations

### 4. Code Size
**Problem**: Each instantiation adds to binary size.
- `List<Point>`, `List<Player>`, `List<Enemy>` = 3x code

**Mitigation**: Link-time optimization, template deduplication

### 5. Debugging
**Problem**: Generic code is harder to debug.
- Mangled names in stack traces
- Multiple copies of same logic

**Mitigation**: Preserve source mappings, better debug info

---

## Recommendations

### For NanoLang Today (December 2025)

**Don't implement full generics yet.** Reasons:

1. **Premature**: Most users aren't hitting List limitations
2. **Costly**: 12-18 months of focused effort
3. **Risky**: High chance of introducing subtle bugs
4. **Complex**: Requires expertise in type theory

### Alternative: Incremental Improvements

#### Short Term (3-6 months)
1. ✅ **Better List ergonomics**: Syntax sugar for `List.new<T>()` instead of `list_T_new`
2. ✅ **More built-in generics**: `Option<T>`, `Result<T, E>`, `HashMap<K, V>`
3. ✅ **Better error messages**: Show which List instantiations are used

#### Medium Term (6-12 months)
1. **Generic functions only** (no inference): Unlock 80% of use cases
2. **Better monomorphization**: Cache instantiations, faster builds
3. **Type aliases**: `type IntList = List<int>`

#### Long Term (12+ months)
1. Consider full generics **only if**:
   - Self-hosted compiler is complete
   - Standard library is mature
   - User base is hitting real limitations
   - Team has compiler expertise

---

## Conclusion

**Full generics are a 12-18 month project** requiring deep compiler expertise. The current `List<T>` approach works well for most use cases.

**Recommended path:**
1. **Phase 1 only**: Generic functions with explicit type args (3 months)
2. **Wait and see**: Does the user base need more?
3. **If yes**: Phase 2 (structs), Phase 3 (enums)
4. **If no**: Invest effort elsewhere (tooling, stdlib, optimizations)

**Don't underestimate the complexity.** Generics touch every part of the compiler:
- Parser (syntax)
- Type checker (substitution, inference, constraints)
- Transpiler (monomorphization, name mangling)
- Runtime (instantiation, caching)
- Tooling (LSP, debugger, profiler)

**Consider alternatives:**
- Code generation tools (generate List_T manually)
- Macro system (textual substitution)
- External preprocessor

The cost/benefit ratio for full generics is **currently unfavorable** for NanoLang.

---

## Appendix: Minimal Working Example

If you do Phase 1 (generic functions), here's what users could write:

```nano
// Generic identity function
fn identity<T>(x: T) -> T {
    return x
}

// Generic map for List<T>
fn map<T, U>(f: fn(T) -> U, xs: List<T>) -> List<U> {
    let result: List<U> = (list_U_new)
    let mut i: int = 0
    while (< i (list_T_length xs)) {
        let x: T = (list_T_get xs i)
        let y: U = (f x)
        (list_U_push result y)
        (set i (+ i 1))
    }
    return result
}

// Usage (explicit instantiation)
fn double(x: int) -> int { return (* x 2) }

fn main() -> int {
    let numbers: List<int> = (list_int_new)
    (list_int_push numbers 1)
    (list_int_push numbers 2)
    (list_int_push numbers 3)
    
    let doubled: List<int> = (map<int, int> double numbers)
    return (list_int_get doubled 0)  // Returns 2
}
```

**This alone would be hugely valuable** and is achievable in 3 months.


