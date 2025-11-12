# Self-Hosting Requirements for nanolang

**Date**: November 12, 2025  
**Status**: Planning Document

## Overview

This document outlines the additional features nanolang needs to implement itself - the path to **self-hosting**. Currently, the nanolang compiler is implemented in C (~3,200 lines). To rewrite it in nanolang, we need to add features that enable compiler construction.

## Current State Analysis

### ✅ What We Have

**Core Language Features:**
- Static typing (int, float, bool, string, void)
- Arrays with bounds checking
- Functions with parameters and return types
- Control flow (if/else, while, for)
- Variables (immutable by default, mutable with `mut`)
- Prefix notation (eliminates precedence issues)
- Shadow tests (mandatory testing)

**Standard Library (24 functions):**
- I/O: `print`, `println`, `assert`
- Math: `abs`, `min`, `max`, `sqrt`, `pow`, `floor`, `ceil`, `round`, `sin`, `cos`, `tan`
- String: `str_length`, `str_concat`, `str_substring`, `str_contains`, `str_equals`
- Array: `at`, `array_length`, `array_new`, `array_set`
- OS: `getcwd`, `getenv`, `range`

### ❌ What We're Missing

To write a compiler in nanolang, we need:

1. **Compound Data Types** - Structs to represent tokens, AST nodes, symbols
2. **Dynamic Data Structures** - Growing lists, hash tables for symbol tables
3. **File I/O** - Read source files, write C output
4. **Advanced String Operations** - Character access, parsing, formatting
5. **Error Handling** - Graceful error propagation without crashing
6. **Module System** - Split compiler across multiple files
7. **System Calls** - Execute C compiler, manage processes
8. **Enums/Sum Types** - Represent token types, AST node types
9. **Pointers/References** - Build linked data structures
10. **Generic Data Structures** - Reusable containers

---

## Priority 1: Essential Compiler Features

These features are **absolutely required** to write any compiler:

### 1.1 Structs (Records)

**Why needed:** Represent complex data like tokens, AST nodes, and symbols.

**Proposed Syntax:**

```nano
# Define a struct type
struct Token {
    type: TokenType,
    value: string,
    line: int,
    column: int
}

# Create an instance
let tok: Token = Token {
    type: TOKEN_NUMBER,
    value: "42",
    line: 1,
    column: 5
}

# Access fields
let line_num: int = tok.line
```

**Why this design:**
- Clear field declarations with types
- Familiar syntax from other languages
- Type-safe field access
- Immutable by default (use `mut` for mutable structs)

**Implementation complexity:** Medium
- Parser needs to handle struct definitions
- Type checker needs struct type tracking
- Transpiler generates C structs

---

### 1.2 Enums (Sum Types)

**Why needed:** Represent token types, AST node types, operator types, etc.

**Proposed Syntax:**

```nano
# Simple enum (like C enum)
enum TokenType {
    TOKEN_NUMBER,
    TOKEN_STRING,
    TOKEN_IDENTIFIER,
    TOKEN_LPAREN,
    TOKEN_RPAREN
}

# Enum with associated data (tagged unions)
enum ASTNode {
    Number(int),
    String(string),
    BinaryOp(string, ASTNode, ASTNode),
    Call(string, array<ASTNode>)
}

# Pattern matching
fn eval_node(node: ASTNode) -> int {
    match node {
        Number(n) => return n,
        BinaryOp("+", left, right) => {
            return (+ (eval_node left) (eval_node right))
        },
        _ => return 0
    }
}
```

**Why this design:**
- Simple enums for basic discriminated types
- Tagged unions for complex AST nodes
- Pattern matching for safe enum handling
- Prevents invalid state representations

**Implementation complexity:** High
- Requires pattern matching
- Memory layout challenges
- Type checker complexity increases

**Alternative (simpler):** Start with C-style enums only (just named integers):

```nano
enum TokenType {
    TOKEN_NUMBER = 0,
    TOKEN_STRING = 1,
    TOKEN_LPAREN = 2
}

# Use with if/else
if (== tok.type TOKEN_NUMBER) {
    # Handle number token
}
```

---

### 1.3 Dynamic Arrays (Vectors/Lists)

**Why needed:** Store variable-length lists of tokens, AST nodes, statements, etc.

**Proposed Syntax:**

```nano
# Dynamic array type
let mut tokens: list<Token> = (list_new)

# Append items (list grows automatically)
(list_push tokens tok1)
(list_push tokens tok2)

# Access by index (bounds checked)
let first: Token = (list_get tokens 0)

# Iterate
let len: int = (list_length tokens)
for i in (range 0 len) {
    let tok: Token = (list_get tokens i)
    (process_token tok)
}
```

**Required Operations:**
- `list_new<T>() -> list<T>` - Create empty list
- `list_push<T>(list: mut list<T>, item: T) -> void` - Append item
- `list_get<T>(list: list<T>, index: int) -> T` - Get item (bounds checked)
- `list_set<T>(list: mut list<T>, index: int, item: T) -> void` - Set item
- `list_length<T>(list: list<T>) -> int` - Get length
- `list_capacity<T>(list: list<T>) -> int` - Get capacity
- `list_clear<T>(list: mut list<T>) -> void` - Remove all items

**Why this design:**
- Arrays are fixed-size, lists are dynamic
- Generic type parameter ensures type safety
- Familiar API from other languages
- Automatic memory management (grows as needed)

**Implementation complexity:** Medium
- Requires generics (or specialized versions)
- Memory management (realloc)
- Transpiles to custom C struct with capacity tracking

---

### 1.4 File I/O

**Why needed:** Read source files, write generated C code.

**Proposed Syntax:**

```nano
# Read entire file as string
fn read_source_file(path: string) -> string {
    let contents: string = (file_read path)
    return contents
}

# Write string to file
fn write_output(path: string, code: string) -> void {
    (file_write path code)
}

# Check if file exists
fn file_exists(path: string) -> bool {
    return (file_exists path)
}
```

**Required Operations:**
- `file_read(path: string) -> string` - Read entire file as string (returns empty on error)
- `file_write(path: string, content: string) -> void` - Write string to file
- `file_exists(path: string) -> bool` - Check if file exists
- `file_append(path: string, content: string) -> void` - Append to file

**Error handling approach:**
- Initially: Return empty string on read failure (check with `str_length`)
- Later: Add proper error handling with Result types

**Implementation complexity:** Low
- Transpiles to C `fopen`, `fread`, `fwrite`, `fclose`
- Add to standard library (similar to `getcwd`)

---

### 1.5 Advanced String Operations

**Why needed:** Character-by-character parsing for lexer.

**Proposed Syntax:**

```nano
# Get character at index
let first_char: string = (str_char_at "Hello" 0)  # "H"

# Character code (ASCII/UTF-8)
let code: int = (str_char_code "A")  # 65

# Create string from character code
let char: string = (str_from_code 65)  # "A"

# String formatting (for code generation)
let code: string = (str_format "int {0} = {1};" "x" "42")
# Result: "int x = 42;"

# Split string by delimiter
let parts: array<string> = (str_split "a,b,c" ",")
# ["a", "b", "c"]
```

**Required Operations:**
- `str_char_at(s: string, index: int) -> string` - Get single character as string
- `str_char_code(s: string) -> int` - Get ASCII/UTF-8 code of first character
- `str_from_code(code: int) -> string` - Create single-char string from code
- `str_format(template: string, ...args) -> string` - Format string with placeholders
- `str_split(s: string, delimiter: string) -> array<string>` - Split by delimiter
- `str_join(parts: array<string>, separator: string) -> string` - Join array into string
- `str_to_int(s: string) -> int` - Parse integer (0 on failure)
- `str_to_float(s: string) -> float` - Parse float (0.0 on failure)

**Implementation complexity:** Low to Medium
- Most transpile to C standard library functions
- `str_format` is more complex (varargs or fixed number of args)

---

## Priority 2: Quality-of-Life Compiler Features

These make compiler implementation much easier but aren't strictly required:

### 2.1 Hash Tables (Dictionaries)

**Why needed:** Fast symbol lookup in symbol tables.

**Proposed Syntax:**

```nano
# Create a hash table
let mut symbols: dict<string, Symbol> = (dict_new)

# Insert key-value pair
(dict_set symbols "x" my_symbol)

# Lookup (returns default value if not found)
let sym: Symbol = (dict_get symbols "x" default_symbol)

# Check if key exists
let exists: bool = (dict_has symbols "x")

# Remove key
(dict_remove symbols "x")
```

**Required Operations:**
- `dict_new<K, V>() -> dict<K, V>` - Create empty dictionary
- `dict_set<K, V>(dict: mut dict<K, V>, key: K, value: V) -> void` - Insert/update
- `dict_get<K, V>(dict: dict<K, V>, key: K, default: V) -> V` - Lookup with default
- `dict_has<K, V>(dict: dict<K, V>, key: K) -> bool` - Check existence
- `dict_remove<K, V>(dict: mut dict<K, V>, key: K) -> void` - Remove key
- `dict_keys<K, V>(dict: dict<K, V>) -> list<K>` - Get all keys
- `dict_values<K, V>(dict: dict<K, V>) -> list<V>` - Get all values

**Why this design:**
- O(1) average lookup time (vs O(n) for arrays)
- Essential for efficient symbol tables
- Familiar API from Python, JavaScript, etc.

**Implementation complexity:** High
- Hash function implementation
- Collision resolution
- Memory management
- Generic types

**Alternative (simpler):** Start with linear search in arrays, optimize later.

---

### 2.2 Error Handling (Result Types)

**Why needed:** Graceful error propagation without crashing.

**Proposed Syntax:**

```nano
# Result type (success or error)
enum Result<T> {
    Ok(T),
    Err(string)
}

# Function that can fail
fn parse_number(s: string) -> Result<int> {
    let value: int = (str_to_int s)
    if (== value 0) {
        if (str_equals s "0") {
            return Ok(0)
        } else {
            return Err("Invalid number")
        }
    } else {
        return Ok(value)
    }
}

# Handle result
fn process() -> void {
    let result: Result<int> = (parse_number "42")
    match result {
        Ok(num) => (println num),
        Err(msg) => (println msg)
    }
}
```

**Why this design:**
- Forces explicit error handling
- No exceptions (keeps language simple)
- Type-safe error propagation
- Compiler can verify all errors are handled

**Implementation complexity:** Medium (requires enums + pattern matching)

**Alternative (simpler):** Use return codes and check them explicitly:

```nano
# Return -1 on error, non-negative on success
fn parse_number(s: string) -> int {
    let value: int = (str_to_int s)
    if (== value 0) {
        if (not (str_equals s "0")) {
            return -1  # Error code
        }
    }
    return value
}
```

---

### 2.3 Module System

**Why needed:** Split compiler into multiple files (lexer, parser, type checker, etc.)

**Proposed Syntax:**

```nano
# File: lexer.nano
module Lexer {
    export fn tokenize(source: string) -> list<Token> {
        # ... implementation
    }
    
    # Private function (not exported)
    fn is_whitespace(c: string) -> bool {
        # ...
    }
}

# File: main.nano
import Lexer

fn main() -> int {
    let tokens: list<Token> = (Lexer.tokenize source)
    return 0
}
```

**Required Features:**
- `module Name { ... }` - Define a module
- `export` keyword - Mark functions/structs as public
- `import ModuleName` - Import another module
- Qualified access: `ModuleName.function`

**Why this design:**
- Clear module boundaries
- Explicit exports (nothing is public by default)
- Namespacing prevents collisions
- Single-file compilation units

**Implementation complexity:** High
- Multi-file compilation
- Dependency resolution
- Name resolution across files
- Circular dependency detection

**Alternative (simpler):** Use single-file compilation, split logically with comments.

---

### 2.4 System/Process Execution

**Why needed:** Execute C compiler on generated code.

**Proposed Syntax:**

```nano
# Execute a command and get exit code
fn compile_c_code(c_file: string, output: string) -> int {
    let cmd: string = (str_format "gcc -o {0} {1}" output c_file)
    let exit_code: int = (system cmd)
    return exit_code
}

# Execute and capture output
fn get_gcc_version() -> string {
    let output: string = (system_output "gcc --version")
    return output
}
```

**Required Operations:**
- `system(cmd: string) -> int` - Execute command, return exit code
- `system_output(cmd: string) -> string` - Execute command, capture stdout

**Implementation complexity:** Low
- Transpiles to C `system()` and `popen()`
- Security considerations (command injection)

---

## Priority 3: Advanced Features (Nice to Have)

These would make the compiler better but can be added later:

### 3.1 Generics

**Why needed:** Reusable data structures (list, dict, option, result).

**Proposed Syntax:**

```nano
# Generic function
fn identity<T>(x: T) -> T {
    return x
}

# Generic struct
struct Box<T> {
    value: T
}

# Usage
let int_box: Box<int> = Box { value: 42 }
let str_box: Box<string> = Box { value: "hello" }
```

**Implementation complexity:** Very High
- Type parameter tracking
- Monomorphization (generate specialized versions)
- Type inference for generics
- Constraint system (trait bounds)

**Alternative:** Generate specialized versions manually or use `any` type (unsafe).

---

### 3.2 Pointers/References

**Why needed:** Build linked data structures (trees, linked lists).

**Proposed Syntax:**

```nano
# Struct with pointer to another node
struct TreeNode {
    value: int,
    left: ptr<TreeNode>,
    right: ptr<TreeNode>
}

# Create node
let mut node: TreeNode = TreeNode {
    value: 42,
    left: null,
    right: null
}

# Take address
let node_ptr: ptr<TreeNode> = (ref node)

# Dereference
let val: int = (deref node_ptr).value
```

**Implementation complexity:** Very High
- Memory safety concerns
- Null pointer handling
- Ownership/borrowing rules
- Manual memory management

**Alternative:** Use arrays with indices instead of pointers (slower but safer).

---

### 3.3 Traits/Interfaces

**Why needed:** Abstract over types, define protocols.

**Example:**

```nano
trait Printable {
    fn to_string(self) -> string
}

# Implement for int
impl Printable for int {
    fn to_string(self) -> string {
        # ... convert int to string
    }
}

# Generic function with trait bound
fn print_value<T: Printable>(value: T) -> void {
    (println (value.to_string))
}
```

**Implementation complexity:** Very High
- Trait resolution
- Method dispatch
- Coherence checking
- Associated types

**Alternative:** Use function pointers or manual dispatch.

---

## Implementation Roadmap

### Phase 1: Minimal Self-Hosting (6-8 months)

**Goal:** Write a basic nanolang compiler in nanolang that can compile itself.

**Required Features:**
1. ✅ Structs (Priority 1.1)
2. ✅ Simple enums (C-style, Priority 1.2)
3. ✅ Dynamic lists (Priority 1.3)
4. ✅ File I/O (Priority 1.4)
5. ✅ Advanced string ops (Priority 1.5)
6. ✅ System execution (Priority 2.4)

**Development approach:**
- Add one feature at a time
- Write shadow tests for each feature
- Update existing compiler (in C) to support new features
- Start writing compiler components in nanolang
- Bootstrap once enough features exist

**Example: Lexer in nanolang**

```nano
struct Token {
    type: int,  # Using int for enum initially
    value: string,
    line: int,
    column: int
}

fn tokenize(source: string) -> list<Token> {
    let mut tokens: list<Token> = (list_new)
    let mut pos: int = 0
    let len: int = (str_length source)
    
    while (< pos len) {
        let c: string = (str_char_at source pos)
        
        if (str_equals c " ") {
            # Skip whitespace
            set pos (+ pos 1)
        } else {
            if (str_equals c "(") {
                let tok: Token = Token {
                    type: TOKEN_LPAREN,
                    value: "(",
                    line: 1,
                    column: pos
                }
                (list_push tokens tok)
                set pos (+ pos 1)
            } else {
                # Handle other tokens...
                set pos (+ pos 1)
            }
        }
    }
    
    return tokens
}

shadow tokenize {
    let tokens: list<Token> = (tokenize "(+ 1 2)")
    assert (== (list_length tokens) 5)
    
    let first: Token = (list_get tokens 0)
    assert (== first.type TOKEN_LPAREN)
}
```

---

### Phase 2: Improved Self-Hosting (8-12 months)

**Goal:** Make the self-hosted compiler maintainable and feature-complete.

**Additional Features:**
1. Hash tables (Priority 2.1)
2. Result types (Priority 2.2)
3. Module system (Priority 2.3)
4. Tagged union enums (Priority 1.2 advanced)

**Benefits:**
- Faster compilation (hash tables for symbol lookup)
- Better error handling (Result types)
- Modular codebase (easier to maintain)
- Better AST representation (tagged unions)

---

### Phase 3: Advanced Self-Hosting (12-18 months)

**Goal:** Optimize and add advanced features.

**Additional Features:**
1. Generics (Priority 3.1)
2. Pointers (Priority 3.2 - if needed)
3. Traits (Priority 3.3 - if needed)

**Benefits:**
- Reusable generic containers
- More efficient data structures
- Better abstractions

---

## Design Decisions

### Why Not Add Everything at Once?

**Incremental development:**
- Each feature adds complexity
- Test one feature thoroughly before adding next
- Easier to maintain backward compatibility
- Reduces risk of introducing bugs

**Bootstrapping strategy:**
- Keep C compiler as "reference implementation"
- Nanolang compiler compiles alongside C compiler
- Test both compilers produce same output
- Eventually retire C compiler

---

## Estimated Complexity

### Lines of Code (Rough Estimates)

**Current C compiler:** ~3,200 lines

**Nanolang self-hosted compiler (estimated):**
- Without modules: ~5,000-6,000 lines (single file)
- With modules: ~4,000-5,000 lines (split across files)
- With generics: ~3,500-4,500 lines (more reusable code)

**Why more lines?**
- Less pointer arithmetic (more explicit indexing)
- More type annotations (verbosity)
- More explicit error handling
- Shadow tests for every function

**Why fewer lines with advanced features?**
- Generics reduce code duplication
- Better abstractions
- Standard library growth

---

## Feature Comparison: Current vs Required

| Feature | Current Status | Required for Self-Hosting | Priority |
|---------|---------------|---------------------------|----------|
| Structs | ❌ Missing | ✅ Essential | P1 |
| Enums | ❌ Missing | ✅ Essential | P1 |
| Lists (dynamic) | ❌ Missing | ✅ Essential | P1 |
| File I/O | ❌ Missing | ✅ Essential | P1 |
| String ops (advanced) | ⚠️ Partial | ✅ Essential | P1 |
| System execution | ❌ Missing | ✅ Essential | P1 |
| Hash tables | ❌ Missing | ⚠️ Very useful | P2 |
| Result types | ❌ Missing | ⚠️ Very useful | P2 |
| Modules | ❌ Missing | ⚠️ Very useful | P2 |
| Generics | ❌ Missing | ⚠️ Useful | P3 |
| Pointers | ❌ Missing | ⚠️ Maybe | P3 |
| Traits | ❌ Missing | ⚠️ Maybe | P3 |

---

## Alternative Approaches

### Option 1: Minimal Compiler (Recommended)

**Add only P1 features:**
- Structs
- Simple enums (int-based)
- Lists
- File I/O
- String operations
- System execution

**Pros:**
- Fastest path to self-hosting
- Smaller language surface area
- Easier to maintain
- Tests core hypothesis

**Cons:**
- Less elegant code
- Performance limitations (no hash tables)
- Single-file compiler (harder to navigate)

**Estimated time:** 6-8 months

---

### Option 2: Quality Compiler

**Add P1 + P2 features:**
- Everything from P1
- Hash tables
- Result types
- Module system
- Tagged union enums

**Pros:**
- Professional-quality codebase
- Better performance
- Easier to maintain long-term
- Demonstrates language maturity

**Cons:**
- Longer development time
- More complex language
- More features to test and maintain

**Estimated time:** 12-16 months

---

### Option 3: Advanced Compiler

**Add P1 + P2 + P3 features:**
- Everything from P1 and P2
- Generics
- Pointers (maybe)
- Traits (maybe)

**Pros:**
- Highly expressive language
- Maximum code reuse
- Industrial-strength compiler

**Cons:**
- Very long development time
- Language complexity increases significantly
- Harder for beginners to learn
- More implementation bugs

**Estimated time:** 18-24 months

---

## Recommendation

**Start with Option 1: Minimal Compiler**

**Rationale:**
1. Validates that nanolang design works for real projects
2. Quickest path to self-hosting (proves core concept)
3. Can always add P2/P3 features later
4. Maintains language simplicity (key design goal)
5. LLMs can still understand the language easily

**Next steps:**
1. Implement structs (most impactful feature)
2. Add simple enums (enables better type modeling)
3. Implement dynamic lists (essential for token/AST storage)
4. Add file I/O (read source, write output)
5. Expand string operations (character access for lexer)
6. Add system execution (invoke gcc)
7. Begin writing lexer in nanolang
8. Continue with parser, type checker, transpiler

**Timeline:**
- Month 1-2: Structs + enums
- Month 3-4: Lists + file I/O
- Month 5-6: String ops + system execution
- Month 7-8: Write compiler in nanolang
- Month 9-10: Bootstrap and test
- Month 11-12: Bug fixes and optimization

---

## Success Criteria

Self-hosting is successful when:

1. ✅ nanolang compiler (written in nanolang) can compile itself
2. ✅ Bootstrapping process works (compile v1 with C compiler, compile v2 with v1, etc.)
3. ✅ Output binaries from both compilers are functionally equivalent
4. ✅ All shadow tests pass
5. ✅ Performance is acceptable (within 2-3x of C compiler)
6. ✅ All examples still work
7. ✅ Documentation is updated

---

## Summary

**Minimum viable features for self-hosting:**
1. Structs - represent complex data
2. Enums - discriminate types
3. Lists - dynamic collections
4. File I/O - read/write files
5. String operations - parsing and formatting
6. System execution - invoke C compiler

**These 6 features enable building a compiler in nanolang.**

**Development effort:** ~6-8 months for minimal self-hosting

**Key insight:** We don't need everything at once. Start minimal, add features as needed. This aligns with nanolang's philosophy of minimalism and clarity.

---

**Next Document:** `STRUCTS_DESIGN.md` - Detailed design for struct implementation

