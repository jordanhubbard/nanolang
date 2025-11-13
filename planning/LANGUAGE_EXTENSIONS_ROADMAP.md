# Language Extensions Roadmap (v2.0)

**Goal:** Enable Full Self-Hosting (Stage 2)  
**Timeline:** 40-60 hours over 2-3 weeks  
**Target:** v2.0.0 release

---

## Overview

To achieve Stage 2 (full self-hosting), we need to implement three critical language features:

1. **Union Types** - For AST node representation
2. **Generic Types** - For type-safe data structures
3. **File I/O** - For reading source files

---

## Feature 1: Union Types (15-20 hours)

### Goal
Allow expressing discriminated unions for AST and other variant types.

### Syntax Design

```nano
# Define a union type
union ASTNode {
    Number { value: int },
    String { value: string },
    BinOp { left: ASTNode, op: string, right: ASTNode },
    Identifier { name: string }
}

# Creating union values
let num: ASTNode = ASTNode.Number { value: 42 }
let id: ASTNode = ASTNode.Identifier { name: "x" }

# Pattern matching (required for unions)
fn eval(node: ASTNode) -> int {
    match node {
        Number(n) => return n.value,
        String(s) => return 0,  # Error for strings
        BinOp(op) => {
            let left_val: int = (eval op.left)
            let right_val: int = (eval op.right)
            if (== op.op "+") {
                return (+ left_val right_val)
            } else {
                return 0
            }
        },
        Identifier(id) => return 0  # Look up in environment
    }
}
```

### Implementation Tasks

**Phase 1: Parser (4-5 hours)**
- [ ] Add `union` keyword to lexer
- [ ] Parse union definitions with variants
- [ ] Parse variant construction: `TypeName.Variant { ... }`
- [ ] Parse match expressions
- [ ] Create AST nodes for unions

**Phase 2: Type System (5-6 hours)**
- [ ] Add `TYPE_UNION` to type enum
- [ ] Store union variants in environment
- [ ] Type check union construction
- [ ] Type check match expressions (exhaustiveness)
- [ ] Ensure all branches return same type

**Phase 3: Transpiler (4-5 hours)**
- [ ] Generate C tagged union representation
- [ ] Generate variant constructors
- [ ] Generate match statement as switch
- [ ] Handle nested unions

**Phase 4: Testing (2-3 hours)**
- [ ] Unit tests for union parsing
- [ ] Type checking tests
- [ ] Transpilation tests
- [ ] Integration tests with AST

### C Code Generation Strategy

```c
/* Union definition */
typedef enum {
    ASTNODE_TAG_NUMBER,
    ASTNODE_TAG_STRING,
    ASTNODE_TAG_BINOP,
    ASTNODE_TAG_IDENTIFIER
} ASTNode_Tag;

typedef struct {
    ASTNode_Tag tag;
    union {
        struct { int64_t value; } number;
        struct { const char* value; } string;
        struct { 
            struct ASTNode* left; 
            const char* op; 
            struct ASTNode* right; 
        } binop;
        struct { const char* name; } identifier;
    } data;
} ASTNode;
```

---

## Feature 2: Generic Types (10-15 hours)

### Goal
Replace type-specific lists with generic `list<T>` type.

### Syntax Design

```nano
# Generic list type
let numbers: list<int> = (list_new)
(list_push numbers 42)
let first: int = (list_get numbers 0)

# Generic function
fn map<T, U>(lst: list<T>, f: fn(T) -> U) -> list<U> {
    let mut result: list<U> = (list_new)
    let mut i: int = 0
    while (< i (list_length lst)) {
        let item: T = (list_get lst i)
        (list_push result (f item))
        set i (+ i 1)
    }
    return result
}
```

### Implementation Tasks

**Phase 1: Parser (3-4 hours)**
- [ ] Parse generic type parameters: `<T>`, `<T, U>`
- [ ] Parse generic type usage: `list<int>`, `array<string>`
- [ ] Parse generic function definitions
- [ ] Create AST nodes for generics

**Phase 2: Type System (5-6 hours)**
- [ ] Add generic type parameters to environment
- [ ] Implement type substitution for generics
- [ ] Type check generic instantiations
- [ ] Monomorphization strategy (compile separate versions)

**Phase 3: Transpiler (3-4 hours)**
- [ ] Generate monomorphized versions of generic types
- [ ] Name mangling for instantiated types: `list_int`, `list_string`
- [ ] Generate specialized function versions
- [ ] Reuse existing runtime implementations

**Phase 4: Testing (2-3 hours)**
- [ ] Generic list tests
- [ ] Generic function tests
- [ ] Multiple type parameter tests
- [ ] Complex generic scenarios

### Monomorphization Strategy

```nano
# Nanolang code
fn identity<T>(x: T) -> T {
    return x
}

let a: int = (identity 42)
let b: string = (identity "hello")
```

```c
/* Generated C code (monomorphized) */
int64_t identity_int(int64_t x) {
    return x;
}

const char* identity_string(const char* x) {
    return x;
}

int64_t a = identity_int(42LL);
const char* b = identity_string("hello");
```

---

## Feature 3: File I/O (5-10 hours)

### Goal
Enable reading and writing files from nanolang programs.

### Syntax Design

```nano
# File operations
fn read_file(path: string) -> string {
    let handle: int = (file_open path "r")
    if (< handle 0) {
        return ""
    } else {
        let content: string = (file_read_all handle)
        (file_close handle)
        return content
    }
}

fn write_file(path: string, content: string) -> bool {
    let handle: int = (file_open path "w")
    if (< handle 0) {
        return false
    } else {
        (file_write handle content)
        (file_close handle)
        return true
    }
}
```

### Implementation Tasks

**Phase 1: Runtime Functions (2-3 hours)**
- [ ] Implement `file_open(path, mode)` → file descriptor
- [ ] Implement `file_read_all(fd)` → string
- [ ] Implement `file_read_line(fd)` → string
- [ ] Implement `file_write(fd, content)` → bool
- [ ] Implement `file_close(fd)` → void
- [ ] Error handling for file operations

**Phase 2: Integration (2-3 hours)**
- [ ] Add file functions to builtin registry
- [ ] Add to transpiler's builtin mapping
- [ ] Type signatures for file operations
- [ ] Documentation

**Phase 3: Testing (2-3 hours)**
- [ ] File reading tests
- [ ] File writing tests
- [ ] Error handling tests
- [ ] Integration tests (read source, compile, write output)

### C Implementation

```c
/* Runtime functions */
int64_t nl_file_open(const char* path, const char* mode) {
    FILE* f = fopen(path, mode);
    return f ? fileno(f) : -1;
}

const char* nl_file_read_all(int64_t fd) {
    FILE* f = fdopen(fd, "r");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buffer = malloc(size + 1);
    fread(buffer, 1, size, f);
    buffer[size] = '\0';
    return buffer;
}

void nl_file_write(int64_t fd, const char* content) {
    FILE* f = fdopen(fd, "w");
    fprintf(f, "%s", content);
}

void nl_file_close(int64_t fd) {
    FILE* f = fdopen(fd, "r");
    fclose(f);
}
```

---

## Feature 4: String Builder (Optional - 5 hours)

### Goal
Efficient string concatenation for code generation.

### Syntax Design

```nano
fn generate_code() -> string {
    let mut sb: StringBuilder = (sb_new)
    (sb_append sb "int main() {\n")
    (sb_append sb "    return 0;\n")
    (sb_append sb "}\n")
    return (sb_to_string sb)
}
```

### Implementation Tasks

**Phase 1: Runtime (2-3 hours)**
- [ ] Implement `StringBuilder` struct
- [ ] Implement `sb_new()` → StringBuilder
- [ ] Implement `sb_append(sb, str)` → void
- [ ] Implement `sb_to_string(sb)` → string
- [ ] Dynamic growth strategy

**Phase 2: Integration (1-2 hours)**
- [ ] Add to builtin registry
- [ ] Type signatures
- [ ] Documentation

**Phase 3: Testing (1-2 hours)**
- [ ] Basic append tests
- [ ] Large string tests
- [ ] Performance tests

---

## Implementation Order (Recommended)

### Week 1: Union Types
**Days 1-2:** Parser implementation  
**Days 3-4:** Type system implementation  
**Day 5:** Transpiler implementation  
**Day 6:** Testing and debugging  
**Day 7:** Documentation

### Week 2: Generic Types
**Days 1-2:** Parser implementation  
**Days 3-4:** Type system + monomorphization  
**Day 5:** Transpiler implementation  
**Days 6-7:** Testing and integration

### Week 3: File I/O + Polish
**Days 1-2:** File I/O implementation  
**Day 3:** String builder (optional)  
**Days 4-5:** Integration testing  
**Days 6-7:** Documentation + examples

---

## Testing Strategy

### Unit Tests
- Test each feature in isolation
- Parser tests for syntax
- Type checker tests for semantics
- Transpiler tests for correct C generation

### Integration Tests
- AST representation using unions
- Generic lists for AST nodes
- File I/O for reading source files
- End-to-end compiler pipeline

### Validation
- All existing tests must pass
- New tests for each feature
- Performance benchmarks
- Memory leak checks

---

## Success Criteria

### Union Types ✓
- [ ] Can represent AST nodes as unions
- [ ] Pattern matching works correctly
- [ ] Type checking prevents invalid matches
- [ ] Generates efficient C code

### Generic Types ✓
- [ ] Can use `list<T>` for any type
- [ ] Type-safe operations
- [ ] Monomorphization generates correct code
- [ ] No runtime overhead

### File I/O ✓
- [ ] Can read source files
- [ ] Can write generated code
- [ ] Error handling works
- [ ] No resource leaks

---

## Milestone: v2.0.0

**When complete, we can:**
1. Represent AST in nanolang (using unions)
2. Store AST nodes in lists (using generics)
3. Read source files (using file I/O)
4. Begin implementing Stage 2 (parser/typechecker/transpiler in nanolang)

**Timeline:** 2-3 weeks of focused work

**Deliverable:** nanolang v2.0 with language extensions ready for self-hosting

---

## Next Steps

1. **Start with Union Types** (most critical for AST)
2. **Create feature branch:** `feature/union-types`
3. **Implement parser first** (get syntax working)
4. **Iterate on type system** (get semantics right)
5. **Transpile to C** (get code generation working)
6. **Test thoroughly** (ensure correctness)

---

**Ready to begin implementing union types?** 🚀

