# Chapter 1: Getting Started

**Learn how to install NanoLang and write your first program.**

Welcome to NanoLang! This chapter will get you up and running with a working NanoLang installation and your first program.

## 1.1 Installation & Setup

### Prerequisites

Before installing NanoLang, ensure you have:
- A Unix-like operating system (Linux, macOS, FreeBSD, or Windows with WSL2)
- GCC or Clang compiler
- Make build tool
- Git

### Installing NanoLang

Clone the repository and build the compiler:

```bash
git clone https://github.com/jordanhubbard/nanolang.git
cd nanolang
make build
```

This creates the NanoLang compiler at `bin/nanoc`.

### Platform-Specific Notes

**macOS users:** The build works on both Apple Silicon and Intel Macs.

**Windows users:** Install WSL2 (Windows Subsystem for Linux) first:

```powershell
# In PowerShell (as Administrator)
wsl --install -d Ubuntu
```

Then follow the Linux instructions inside WSL.

### Verifying Installation

Check that the compiler is working:

```bash
./bin/nanoc --version
```

You should see version information printed.

## 1.2 Your First NanoLang Program

Let's write the traditional "Hello, World!" program.

Create a file named `hello.nano`:

```nano
fn main() -> int {
    (println "Hello, World!")
    return 0
}

shadow main {
    assert true
}
```

### Understanding the Syntax

Let's break down each part:

- `fn main() -> int` - Declares a function named `main` that returns an integer
- `(println "Hello, World!")` - Calls the `println` function using prefix notation
- `return 0` - Returns 0 to indicate success
- `shadow main { ... }` - A test block that runs at compile time

**Key Points:**
- Every program needs a `main()` function that returns `int`
- Function calls use prefix notation: `(function arg1 arg2)`
- Shadow tests are mandatory for all functions (except `main` can have a trivial test)

### Compiling and Running

Compile the program:

```bash
./bin/nanoc hello.nano -o hello
```

This transpiles your NanoLang code to C and compiles it to a native binary.

Run it:

```bash
./hello
```

Output:
```
Hello, World!
```

## 1.3 Hello World Walkthrough

Let's write a more interesting version that demonstrates NanoLang's core features.

```nano
fn greet(name: string) -> string {
    return (+ "Hello, " name)
}

shadow greet {
    assert (== (greet "World") "Hello, World")
    assert (== (greet "NanoLang") "Hello, NanoLang")
}

fn main() -> int {
    (println (greet "World"))
    (println (greet "NanoLang"))
    return 0
}

shadow main {
    assert true
}
```

### Step-by-Step Explanation

**1. Function Definition**
```nano
fn greet(name: string) -> string {
```
- `fn` declares a function
- `greet` is the function name
- `(name: string)` is a parameter with explicit type annotation
- `-> string` specifies the return type

**2. String Concatenation**
```nano
return (+ "Hello, " name)
```
- String concatenation uses the `+` operator in prefix notation
- `(+ "Hello, " name)` combines two strings

**3. Shadow Tests**
```nano
shadow greet {
    assert (== (greet "World") "Hello, World")
    assert (== (greet "NanoLang") "Hello, NanoLang")
}
```
- `shadow` declares a test block for the `greet` function
- Tests run at compile time
- `assert` checks that conditions are true
- `(== a b)` tests equality using prefix notation

**4. Calling Functions**
```nano
(println (greet "World"))
```
- Function calls are nested: inner calls execute first
- `(greet "World")` returns `"Hello, World"`
- `(println ...)` prints the result

### Running the Program

```bash
./bin/nanoc hello.nano -o hello
./hello
```

Output:
```
Hello, World
Hello, NanoLang
```

## 1.4 Compilation & Execution

### The Compilation Process

NanoLang is a **transpiled language**. When you compile a `.nano` file:

1. **Lexing & Parsing** - Reads your code and builds an Abstract Syntax Tree (AST)
2. **Type Checking** - Verifies types are correct
3. **Shadow Tests** - Runs compile-time tests
4. **Transpilation** - Converts NanoLang to C code
5. **C Compilation** - Uses GCC/Clang to create a native binary

### Compiler Flags

**Basic compilation:**
```bash
./bin/nanoc input.nano -o output
```

**Keep generated C code:**
```bash
./bin/nanoc input.nano -o output --keep-c
```

The generated C file will be in the same directory as your output.

**Show intermediate code:**
```bash
./bin/nanoc input.nano -o output -fshow-intermediate-code
```

Prints the generated C to stdout (useful for debugging).

**Verbose output:**
```bash
./bin/nanoc input.nano -o output --verbose
```

Shows detailed compilation steps.

### Troubleshooting Compilation Errors

#### Missing Shadow Test

**Error:**
```
Warning: Function 'my_function' is missing a shadow test
```

**Fix:** Add a shadow test block:
```nano
shadow my_function {
    assert (== (my_function 5) expected_result)
}
```

#### Type Mismatch

**Error:**
```
Type error: Expected 'int' but got 'string'
```

**Fix:** Check type annotations and function calls. NanoLang requires explicit types.

#### Syntax Error

**Error:**
```
Parse error at line 5: unexpected token
```

**Fix:** Check that you're using prefix notation correctly. Remember:
- ✅ `(+ 1 2)` not `1 + 2`
- ✅ `(println "hi")` not `println("hi")`

### Example: Complete Program

Here's a complete program with functions, types, and shadow tests:

```nano
fn add(a: int, b: int) -> int {
    return (+ a b)
}

shadow add {
    assert (== (add 2 3) 5)
    assert (== (add 0 0) 0)
    assert (== (add -1 1) 0)
}

fn multiply(a: int, b: int) -> int {
    return (* a b)
}

shadow multiply {
    assert (== (multiply 3 4) 12)
    assert (== (multiply 0 5) 0)
    assert (== (multiply -2 3) -6)
}

fn calculate(x: int, y: int) -> int {
    let sum: int = (add x y)
    let product: int = (multiply x y)
    return (+ sum product)
}

shadow calculate {
    assert (== (calculate 2 3) 11)  # 2+3=5, 2*3=6, 5+6=11
}

fn main() -> int {
    let result: int = (calculate 5 7)
    (println (int_to_string result))
    return 0
}

shadow main {
    assert true
}
```

Save this as `calculate.nano`, compile, and run:

```bash
./bin/nanoc calculate.nano -o calculate
./calculate
```

Output:
```
47
```

### Next Steps

Now that you have NanoLang installed and understand the basics, you're ready to learn:
- [Chapter 2: Basic Syntax & Types](02_syntax_types.md) - Master prefix notation and type system
- [Chapter 3: Variables & Bindings](03_variables.md) - Learn about immutability and mutation
- [Chapter 4: Functions](04_functions.md) - Deep dive into functions and shadow tests

### Common First-Time Questions

**Q: Why prefix notation?**  
A: Prefix notation eliminates ambiguity. There's exactly ONE way to write each operation, making code generation by LLMs more reliable.

**Q: What if my shadow tests fail?**  
A: The compiler will show you which assertion failed. Fix your function or your test.

**Q: Can I skip shadow tests?**  
A: No. Shadow tests are mandatory (except for `extern` functions calling C code). They're a core design feature of NanoLang.

**Q: How do I debug my program?**  
A: Start with shadow tests to verify each function works correctly. Use `(println ...)` to print values. See [Chapter 13: log module](../part3_modules/13_text_processing/log.md) for structured logging.

**Q: Where are the standard library functions?**  
A: See [Chapter 9: Core Utilities](../part2_stdlib/09_core_utilities.md) for built-in functions.

---

**Previous:** [User Guide Index](../index.md)  
**Next:** [Chapter 2: Basic Syntax & Types](02_syntax_types.md)
