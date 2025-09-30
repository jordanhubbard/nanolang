# nanolang
A tiny silly language meant to be targeted by coding LLMs 

## Overview

NanoLang is a minimalist programming language with a simple syntax designed to be easy for LLMs to generate and understand. The core interpreter is implemented in C, with a focus on clarity and simplicity.

## Features

- **Simple Syntax**: Minimal keywords and straightforward grammar
- **Dynamic Typing**: Supports integers and booleans
- **Variables**: Let bindings and assignments
- **Arithmetic**: Addition, subtraction, multiplication, division
- **Comparisons**: Equality, less than, greater than
- **Control Flow**: If/else statements and while loops
- **Interactive REPL**: Read-Eval-Print-Loop for interactive programming

## Building

Build the interpreter using the provided Makefile:

```bash
make
```

This creates the `nano` executable.

## Running

### Interactive Mode (REPL)

Start the interactive interpreter:

```bash
./nano
```

Example session:
```
nano> let x = 10;
nano> let y = 20;
nano> print x + y;
30
nano> exit
```

### File Execution

Run a NanoLang program from a file:

```bash
./nano examples/fibonacci.nano
```

## Language Syntax

### Variables

Declare variables with `let`:
```javascript
let x = 42;
let message = 100;
```

Reassign existing variables:
```javascript
x = x + 1;
```

### Arithmetic Expressions

```javascript
let a = 10 + 5;    // Addition
let b = 20 - 8;    // Subtraction
let c = 3 * 4;     // Multiplication
let d = 15 / 3;    // Division
```

### Comparisons

```javascript
let x = 5;
let isEqual = x == 5;      // true
let isLess = x < 10;       // true
let isGreater = x > 3;     // true
```

### Printing

```javascript
print 42;
print x + y;
```

### Control Flow

If statements:
```javascript
if (x > 10) {
    print 1;
} else {
    print 0;
}
```

While loops:
```javascript
let i = 0;
while (i < 10) {
    print i;
    i = i + 1;
}
```

### Comments

Single-line comments start with `//`:
```javascript
// This is a comment
let x = 42;  // This is also a comment
```

## Examples

See the `examples/` directory for sample programs:

- `variables.nano` - Variable declarations and arithmetic
- `conditionals.nano` - If/else statements
- `fibonacci.nano` - Fibonacci sequence generator
- `factorial.nano` - Factorial calculation

## Language Specification

The formal type system and operation definitions are in `spec.json`, following JSON Schema format. This specification defines:

- **Primitive types**: i32, i64, bool
- **Operations**: add, sub, mul, div, eq, lt, gt
- **Runtime functions**: print, input

## Development

Clean build artifacts:
```bash
make clean
```

Run tests:
```bash
make test
```

## License

Apache License 2.0 - See LICENSE file for details
 
