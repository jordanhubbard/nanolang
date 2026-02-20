# My Design: A Self-Reflection

I have reached a point where I can reflect on my own design. I have compiled myself, I have proved my core correct, and I have run my own test suite. This is how I evaluate the decisions that made me.

---

## 1. Syntax and Semantics

### 1.1 Notation
I use prefix notation for function calls and support both prefix and infix for binary operators.

```nano
# Prefix notation
(+ a b)
(* (+ x 1) (- y 2))
(println "hello")

# Infix notation
a + b
(x + 1) * (y - 2)
```

I chose this because it eliminates operator precedence confusion. In my world, all infix operators have equal precedence and evaluate left-to-right. If you want `a * (b + c)`, you must write the parentheses. This makes my AST construction unambiguous and ensures that what you see is exactly what I execute. I find this consistency more valuable than memorizing a precedence table.

### 1.2 Mandatory Shadow Tests
I require a shadow test for every function. I will not compile a function that hasn't been asked to prove itself.

```nano
fn double(x: int) -> int {
    return (* x 2)
}

shadow double {
    assert (== (double 5) 10)
}
```

This is my core mechanism for honesty. It forces you to write executable specifications and gives immediate feedback. I acknowledge that this can feel heavy for trivial functions, and I allow empty `shadow main {}` blocks for entry points. I plan to add more utilities like `setup` and `teardown` to make this even more effective.

### 1.3 Explicit Typing and Inference
I require explicit types for function signatures and global variables, but I can infer local variable types.

```nano
let x = 42               # I infer int
fn add(a: int, b: int) -> int { ... }
```

I refuse to use complex, multi-level type inference that produces cryptic error messages. By keeping function signatures explicit, I ensure that my code remains self-documenting and my error reporting stays direct.

---

## 2. Type System

### 2.1 Primitive Types
I support `int`, `float`, `bool`, `string`, and `void`. I also have `char` for UTF-8 characters and sized integers like `i8`, `u32`, and `usize`. This covers my current needs, though I recognize that more specialized types are sometimes necessary for low-level systems work.

### 2.2 Composite Types
I have `struct`, `enum`, `union`, `array<T>`, and `tuple`.

My structs are straightforward and functional:
```nano
struct Point {
    x: int
    y: int
}
```

My enums and unions are a good start, and I have implemented `Result<T, E>` and `Option<T>` patterns to handle errors and optional values without resorting to null. I am working on making my pattern matching more exhaustive to catch unhandled cases at compile time.

---

## 3. Memory Management

### 3.1 Automatic Reference Counting (ARC)
I use zero-overhead reference counting for memory management. I do not have a separate garbage collection phase that stops the world; I release memory as soon as the last reference to it is gone. This makes my performance predictable and my memory usage deterministic.

### 3.2 Mutability
I am immutable by default. You must use the `mut` keyword to change a value.

```nano
let x: int = 42          # Immutable
let mut y: int = 0       # Mutable
set y (+ y 1)
```

This encourages a functional style and makes side effects visible. I consider this one of my strongest design choices.

---

## 4. Module and Namespace System

I have evolved from a global namespace to a structured module system.

```nano
module my_app
from "modules/std/io/stdio.nano" import fopen, fclose
from "modules/graphics.nano" import sqrt as gfx_sqrt
```

I support explicit imports, private symbols, and module aliasing. This prevents name collisions and makes my boundaries clear. I use the `pub` keyword to define my public API. While I am still refining my circular dependency detection, my current system is stable for large projects.

---

## 5. Standard Library and FFI

I provide a core library for math, strings, arrays, I/O, and OS operations. I have added collections like `List<T>` and I am working on more advanced structures.

My FFI strategy is direct and safe. I use `extern fn` for C calls and can isolate them in a separate process (the co-process model) so that a crash in an external library does not take me down. I require `unsafe {}` blocks for these calls because I am honest about the danger of leaving my verified world.

---

## 6. Error Handling

I have moved beyond silent failures. I now use the `Result<T, E>` pattern for structured error handling.

```nano
enum Result<T, E> {
    Ok(T)
    Err(E)
}

match (read_file "test.txt") {
    Ok(content) => (println content)
    Err(e) => (println "Error reading file")
}
```

This is the correct approach for a systems language. It avoids the overhead of exceptions while forcing you to handle the possibility of failure.

---

## 7. Concurrency

I do not yet have a native concurrency model. This is a gap I am addressing. I am evaluating green threads versus an async/await model. Until then, I rely on process-level isolation and FFI-based threading for concurrent tasks.

---

## 8. My Own Assessment

I evaluate myself across these categories. I am honest about where I stand.

| Category | Status | Notes |
|----------|-------|-------|
| **Syntax Design** | High | Consistent, unambiguous, and machine-friendly. |
| **Type System** | Medium | Strong basics, including Result/Option, but needs more polymorphism. |
| **Memory Model** | High | ARC provides safety with predictable performance. |
| **Module System** | Medium | Namespaces work, but visibility controls could be tighter. |
| **Standard Library** | Medium | Essential tools are there; advanced collections are coming. |
| **Error Handling** | Medium | Result type exists; I need to use it more consistently in my own core. |
| **Concurrency** | Low | Not yet implemented natively. |
| **Performance** | High | Native C transpilation and NanoISA both perform well. |

---

## 9. My Priorities

I have a clear path forward. I do not need external advice to know what matters most to me:

1. **Native Concurrency**: I need a model for threads or fibers that fits my safety goals.
2. **Exhaustive Pattern Matching**: I want to prove that every case in a `match` expression is handled.
3. **Trait System**: I need a way to define shared behavior without sacrificing my simplicity.
4. **Enhanced Tooling**: I need a language server and a standard formatter to help others speak my language.

I am a work in progress, but I am built on a foundation of proof and testing. I say what I mean, and I hold myself accountable.
