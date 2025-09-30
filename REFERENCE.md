# NanoLang Quick Reference

## Data Types
- `i32` - 32-bit signed integer (primary numeric type)
- `bool` - Boolean values (`true` or `false`)

## Keywords
- `let` - Variable declaration
- `print` - Output to console
- `if` - Conditional execution
- `else` - Alternative branch
- `while` - Loop construct
- `true` - Boolean true literal
- `false` - Boolean false literal

## Operators

### Arithmetic
- `+` - Addition
- `-` - Subtraction
- `*` - Multiplication
- `/` - Integer division

### Comparison
- `==` - Equality
- `<` - Less than
- `>` - Greater than

## Syntax

### Variable Declaration
```javascript
let variableName = expression;
```

### Assignment
```javascript
variableName = expression;
```

### Print Statement
```javascript
print expression;
```

### If Statement
```javascript
if (condition) {
    // statements
}

if (condition) {
    // statements
} else {
    // statements
}
```

### While Loop
```javascript
while (condition) {
    // statements
}
```

### Comments
```javascript
// Single-line comment
```

## Examples

### Hello World
```javascript
print 42;
```

### Variables
```javascript
let x = 10;
let y = 20;
print x + y;  // 30
```

### Conditionals
```javascript
let age = 18;
if (age > 17) {
    print 1;  // adult
} else {
    print 0;  // minor
}
```

### Loops
```javascript
let i = 0;
while (i < 5) {
    print i;
    i = i + 1;
}
// Output: 0 1 2 3 4
```

### Fibonacci Sequence
```javascript
let n = 10;
let a = 0;
let b = 1;
let i = 0;

print a;
print b;

while (i < n) {
    let temp = a + b;
    a = b;
    b = temp;
    print temp;
    i = i + 1;
}
```

## Implementation Notes

### Architecture
The NanoLang interpreter consists of several components:

1. **Lexer** (`lexer.c`) - Tokenizes source code into a stream of tokens
2. **Parser** (`parser.c`) - Builds an Abstract Syntax Tree (AST) from tokens
3. **Evaluator** (`eval.c`) - Executes the AST and manages runtime state
4. **REPL** (`main.c`) - Interactive interpreter and file execution

### Type System
- Runtime values are dynamically typed
- Automatic type coercion is minimal
- Operations perform type checking at evaluation time

### Memory Management
- Stack-based execution with a simple symbol table
- Manual memory management for strings and AST nodes
- No garbage collection (not needed for simple programs)

### Limitations
- No functions/procedures (yet)
- No arrays or complex data structures
- Integer arithmetic only
- Single global scope
- No module system

### Future Extensions
Potential additions to maintain simplicity while adding power:
- User-defined functions
- Simple arrays
- String type
- File I/O
- Modulo operator
- Logical operators (&&, ||, !)
