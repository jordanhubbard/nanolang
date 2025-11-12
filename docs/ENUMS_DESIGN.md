# Enums Design for nanolang

**Status:** Design Phase  
**Priority:** #2 for Self-Hosting (after structs)  
**Principles:** Safety, Simplicity, Type-Safe Constants

## Overview

Enums provide type-safe named constants. Essential for representing token types, AST node types, and other discriminated values in the compiler.

## Design Philosophy

**Start Simple:** C-style enums only (just named integers)
- No tagged unions initially (complex, requires pattern matching)
- No associated data (can add later if needed)
- Just: name → integer mapping

**Why Simple First:**
-

Easier to implement
- Sufficient for most compiler needs
- Can be extended later
- Maintains language minimalism

## Syntax Design

### Enum Declaration

```nano
enum EnumName {
    VARIANT1,
    VARIANT2,
    VARIANT3
}
```

**With explicit values:**

```nano
enum TokenType {
    TOKEN_NUMBER = 0,
    TOKEN_STRING = 1,
    TOKEN_LPAREN = 2,
    TOKEN_RPAREN = 3
}
```

**Rules:**
- `enum` keyword introduces declaration
- Variants are UPPERCASE by convention (not enforced)
- Can specify explicit integer values
- If not specified, auto-increment from 0
- Enums are just named integers (type alias)

---

### Using Enum Values

```nano
enum TokenType {
    TOKEN_NUMBER,
    TOKEN_STRING,
    TOKEN_LPAREN
}

fn main() -> int {
    let t: int = TOKEN_NUMBER  # t = 0
    
    if (== t TOKEN_NUMBER) {
        print "It's a number token"
    }
    
    return 0
}
```

**Rules:**
- Enum variants are treated as `int` constants
- Can use in comparisons, assignments, etc.
- No special enum type (keeps implementation simple)
- Type checker knows about enum definitions

---

## Examples

### Example 1: Token Types

```nano
enum TokenType {
    TOKEN_NUMBER = 0,
    TOKEN_STRING = 1,
    TOKEN_IDENTIFIER = 2,
    TOKEN_LPAREN = 3,
    TOKEN_RPAREN = 4,
    TOKEN_LBRACE = 5,
    TOKEN_RBRACE = 6,
    TOKEN_PLUS = 7,
    TOKEN_MINUS = 8
}

struct Token {
    type: int,      # Store enum value
    value: string,
    line: int
}

fn is_operator(tok_type: int) -> bool {
    if (== tok_type TOKEN_PLUS) {
        return true
    } else {
        if (== tok_type TOKEN_MINUS) {
            return true
        } else {
            return false
        }
    }
}

shadow is_operator {
    assert (== (is_operator TOKEN_PLUS) true)
    assert (== (is_operator TOKEN_MINUS) true)
    assert (== (is_operator TOKEN_NUMBER) false)
}
```

---

### Example 2: AST Node Types

```nano
enum ASTNodeType {
    AST_NUMBER,
    AST_STRING,
    AST_BINARY_OP,
    AST_CALL,
    AST_FUNCTION
}

struct ASTNode {
    node_type: int,  # Store enum value
    line: int
    # ... other fields depend on node_type
}

fn is_literal(node_type: int) -> bool {
    if (== node_type AST_NUMBER) {
        return true
    } else {
        if (== node_type AST_STRING) {
            return true
        } else {
            return false
        }
    }
}

shadow is_literal {
    assert (== (is_literal AST_NUMBER) true)
    assert (== (is_literal AST_BINARY_OP) false)
}
```

---

### Example 3: Status Codes

```nano
enum Status {
    STATUS_OK = 0,
    STATUS_ERROR = 1,
    STATUS_WARNING = 2
}

fn process_file(path: string) -> int {
    let exists: bool = (file_exists path)
    if exists {
        return STATUS_OK
    } else {
        return STATUS_ERROR
    }
}

shadow process_file {
    # Test with files we know exist/don't exist
    # (when file_exists is implemented)
}
```

---

## Implementation Plan

### Phase 1: Lexer (Day 1)

**Add new token:**
```c
TOKEN_ENUM  // "enum" keyword
```

**Tasks:**
- [x] Add "enum" to keywords list
- [x] Test tokenization

---

### Phase 2: Parser (Days 1-2)

**Add new AST node:**
```c
typedef enum {
    // ... existing ...
    AST_ENUM_DEF,  // enum definition
} ASTNodeType;

struct {
    char *name;               // Enum name
    char **variant_names;     // Variant names
    int *variant_values;      // Variant values (or NULL for auto)
    int variant_count;        // Number of variants
} enum_def;
```

**Parse enum declaration:**
```nano
enum TokenType {
    TOKEN_NUMBER,      # auto: 0
    TOKEN_STRING = 1,  # explicit: 1
    TOKEN_LPAREN       # auto: 2 (previous + 1)
}
```

**Tasks:**
- [ ] Parse `enum Name { ... }`
- [ ] Handle auto-incrementing values
- [ ] Handle explicit values
- [ ] Validate no duplicate names
- [ ] Write parser tests

---

### Phase 3: Type Checker (Days 2-3)

**Store enum definitions:**
```c
typedef struct {
    char *enum_name;          // Enum name
    char **variant_names;     // Variant names
    int *variant_values;      // Variant values
    int variant_count;
} EnumDef;

// Add to Environment
typedef struct {
    // ... existing ...
    EnumDef *enums;
    int enum_count;
} Environment;
```

**Tasks:**
- [ ] Store enum definitions in environment
- [ ] Check no duplicate enum names
- [ ] Check no duplicate variant names within enum
- [ ] Register variants as int constants
- [ ] Handle in type checking
- [ ] Write tests

**Error messages:**
```
Error: Enum 'TokenType' already defined at line 10
Error: Duplicate variant 'TOKEN_NUMBER' in enum 'TokenType'
Error: Unknown enum variant 'TOKEN_NUMER' (did you mean 'TOKEN_NUMBER'?)
```

---

### Phase 4: Interpreter (Day 3)

**Tasks:**
- [ ] Evaluate enum definitions (register constants)
- [ ] Enum variants work as integer values
- [ ] No special runtime representation needed

**Note:** Enums are compile-time only. At runtime, they're just integers.

---

### Phase 5: C Transpiler (Days 4-5)

**C code generation:**

```nano
enum TokenType {
    TOKEN_NUMBER = 0,
    TOKEN_STRING = 1,
    TOKEN_LPAREN = 2
}
```

Transpiles to:

```c
typedef enum {
    TOKEN_NUMBER = 0,
    TOKEN_STRING = 1,
    TOKEN_LPAREN = 2
} TokenType;

// Also define as macros for compatibility
#define TOKEN_NUMBER 0
#define TOKEN_STRING 1
#define TOKEN_LPAREN 2
```

**Tasks:**
- [ ] Generate C enum typedefs
- [ ] Generate macro definitions
- [ ] Handle auto-incrementing
- [ ] Test generated C compiles

---

### Phase 6: Testing & Documentation (Days 5-6)

**Test files:**
- [ ] `tests/unit/enum_basic.nano` - Basic enum usage
- [ ] `tests/unit/enum_explicit_values.nano` - Explicit values
- [ ] `tests/unit/enum_auto_increment.nano` - Auto-increment
- [ ] `tests/negative/enum_errors/` - Error cases
- [ ] `examples/18_enums.nano` - Example program

**Documentation:**
- [ ] Update SPECIFICATION.md
- [ ] Update QUICK_REFERENCE.md
- [ ] Update GETTING_STARTED.md
- [ ] Update IMPLEMENTATION_STATUS.md

---

## Advanced Features (Future, Not Now)

### Tagged Unions (Later)

```nano
# NOT implementing now, but possible future extension
enum Result {
    Ok(int),      # Associated data
    Err(string)   # Associated data
}

# Would require pattern matching:
match result {
    Ok(value) => print value,
    Err(msg) => print msg
}
```

**Why not now:**
- Complex to implement
- Requires pattern matching
- Not essential for self-hosting
- Can add later if needed

**Alternative for now:**
```nano
# Use separate fields and check enum
struct Result {
    status: int,     # OK or ERR enum value
    int_value: int,  # Only valid if OK
    error_msg: string  # Only valid if ERR
}
```

---

## C Transpilation Examples

### Example 1: Simple Enum

**nanolang:**
```nano
enum Color {
    RED,
    GREEN,
    BLUE
}

fn main() -> int {
    let c: int = RED
    print c
    return 0
}
```

**Generated C:**
```c
typedef enum {
    RED = 0,
    GREEN = 1,
    BLUE = 2
} Color;

#define RED 0
#define GREEN 1
#define BLUE 2

int main(void) {
    int64_t c = RED;
    printf("%lld\n", (long long)c);
    return 0;
}
```

---

## Timeline

**Total Time:** 4-6 weeks (after structs complete)

- **Days 1-2:** Lexer + Parser
- **Days 2-3:** Type checker
- **Days 3-4:** Interpreter
- **Days 4-5:** Transpiler
- **Days 5-6:** Testing + Documentation
- **Extra time:** Buffer for bugs

**Note:** Enums are simpler than structs, so faster to implement.

---

## Success Criteria

✅ All tests pass:
- [ ] Basic enum definition
- [ ] Auto-incrementing values
- [ ] Explicit values
- [ ] Enum values in expressions
- [ ] Type checking
- [ ] Error messages

✅ Can represent token types:
- [ ] TokenType enum works
- [ ] Can use in Token struct
- [ ] Ready for lexer

✅ Documentation complete

---

## Dependencies

**Requires:**
- Structs (to store enum values in structs)

**Unlocks:**
- Better type representation in compiler
- Lexer implementation (token types)
- Parser implementation (AST node types)

---

**Status:** Ready to implement (after structs)  
**Priority:** #2  
**Estimated Time:** 4-6 weeks  
**Dependencies:** Structs

