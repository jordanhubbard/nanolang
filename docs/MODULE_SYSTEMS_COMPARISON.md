# Module Systems Comparison: Python, Go, Ruby, Elixir vs NanoLang

**Purpose:** Understand what makes module systems feel "first-class"  
**Date:** 2025-01-08

---

## What Makes a Module System Feel "First-Class"?

### Key Characteristics

1. **Modules are values** - Can be passed, stored, inspected
2. **Introspection built-in** - List exports, query metadata
3. **Clear safety/trust model** - Know what's safe vs unsafe
4. **Namespace control** - Explicit imports, no pollution
5. **Tooling support** - Docs, linters, analyzers can query modules

---

## Python: Modules as Objects

### Import Syntax

```python
import math                    # Import entire module
from math import sqrt, sin     # Import specific functions
import numpy as np             # Alias
from os import *               # Import all (discouraged)
```

### Module as First-Class Object

```python
import math

# Module IS an object
type(math)  # <class 'module'>

# Access as attribute
math.sqrt(16)  # 4.0

# Introspection
dir(math)  # ['__doc__', '__name__', 'sqrt', 'sin', 'cos', ...]
math.__name__  # 'math'
math.__file__  # '/usr/lib/python3.9/lib-dynload/math.cpython-39.so'
math.__dict__  # Full namespace dictionary

# Check if function exists
hasattr(math, 'sqrt')  # True
hasattr(math, 'foo')   # False

# Get function dynamically
getattr(math, 'sqrt')  # <built-in function sqrt>
```

### Inspection Module

```python
import inspect
import math

# Get all functions
funcs = inspect.getmembers(math, inspect.isfunction)
# [('sqrt', <built-in>), ('sin', <built-in>), ...]

# Function signature
sig = inspect.signature(math.sqrt)
print(sig)  # (x, /)

# Source code (for pure Python modules)
import json
source = inspect.getsource(json.loads)
print(source)  # Full function source
```

### Safety Model

**Problem:** Python has no built-in module safety markers

```python
import ctypes  # Unsafe! Can crash Python
import requests  # Safe HTTP library

# No way to tell which is safe without reading docs
```

**Community Solution:** Type hints + mypy

```python
from typing import List, Dict

def safe_function(items: List[str]) -> Dict[str, int]:
    # Type checker verifies safety
    pass
```

---

## Go: Package-Level Organization

### Import Syntax

```go
package main

import (
    "fmt"              // Standard library
    "math"
    "unsafe"           // Explicitly unsafe!
    "github.com/user/repo"  // Remote package
)
```

### Package Metadata

```go
package mypackage

// Package-level documentation
// This appears in godoc

// Exported (capitalized)
func PublicFunction() {
    privateFunction()  // Can call private
}

// Unexported (lowercase)
func privateFunction() {
    // Only visible within package
}
```

### Introspection via Reflection

```go
import "reflect"

// Get type information
t := reflect.TypeOf(math.Sqrt)
fmt.Println(t)  // func(float64) float64

// List package functions (compile-time via go doc)
// $ go doc math
// package math -- go/doc
// func Sqrt(x float64) float64
// func Sin(x float64) float64
// ...
```

### Safety Model: `unsafe` Package

**Explicit Unsafe Import:**
```go
import "unsafe"

// Using unsafe is VISUALLY OBVIOUS
ptr := unsafe.Pointer(&x)
```

**Compile-Time Checks:**
```bash
# Lint for unsafe usage
$ go vet ./...
# Flag unsafe packages
$ grep -r "import unsafe" .
```

**Why It Works:**
- ✅ One import line makes entire usage visible
- ✅ No `unsafe {}` blocks scattered everywhere
- ✅ Easy to audit: search for "import unsafe"
- ✅ Tooling can flag unsafe packages

---

## Ruby: Modules as Objects

### Import Syntax

```ruby
require 'json'           # Load library
require_relative 'my_module'  # Load local file
```

### Module as Object

```ruby
module Vector2D
  def self.add(v1, v2)
    [v1[0] + v2[0], v1[1] + v2[1]]
  end
  
  def self.magnitude(v)
    Math.sqrt(v[0]**2 + v[1]**2)
  end
end

# Module IS an object
Vector2D.class  # Module

# List methods
Vector2D.methods(false)  # [:add, :magnitude]

# Check method exists
Vector2D.respond_to?(:add)  # true

# Call dynamically
Vector2D.send(:add, [1,2], [3,4])  # [4, 6]
```

### Introspection

```ruby
module MyModule
  @@version = "1.0.0"
  @@safe = true
  
  def self.version
    @@version
  end
  
  def self.safe?
    @@safe
  end
end

# Custom metadata
MyModule.version  # "1.0.0"
MyModule.safe?    # true

# List instance variables
MyModule.instance_variables  # [:@@version, :@@safe]

# Module hierarchy
MyModule.ancestors  # [MyModule]
```

### Refinements (Scoped Monkey-Patching)

```ruby
module StringExtensions
  refine String do
    def shout
      self.upcase + "!"
    end
  end
end

# Only active when explicitly used
using StringExtensions
"hello".shout  # "HELLO!"
```

**Why It Works:**
- ✅ Explicit scope control
- ✅ No global pollution
- ✅ Clear activation point

---

## Elixir: Compile-Time Module Attributes

### Module Definition

```elixir
defmodule Vector2D do
  @moduledoc """
  2D vector mathematics module.
  Provides pure functions for vector operations.
  """
  
  @vsn "1.0.0"
  @safe true
  @author "Elixir Team"
  
  @doc """
  Adds two 2D vectors.
  """
  def add({x1, y1}, {x2, y2}) do
    {x1 + x2, y1 + y2}
  end
  
  def magnitude({x, y}) do
    :math.sqrt(x * x + y * y)
  end
end
```

### Compile-Time Introspection

```elixir
# List functions
Vector2D.__info__(:functions)
# [add: 2, magnitude: 1]

# List attributes
Vector2D.__info__(:attributes)
# [vsn: ["1.0.0"], moduledoc: "...", safe: true]

# Module compiled?
Vector2D.__info__(:compile)
# [...compile options...]

# Full module info
Vector2D.module_info()
# [module: Vector2D, exports: [...], attributes: [...]]
```

### Custom Attributes for Safety

```elixir
defmodule UnsafeFFI do
  @unsafe true
  @ffi_calls [:c_malloc, :c_free, :memcpy]
  
  def allocate(size) do
    # Calls C function
    :c_malloc(size)
  end
end

# Check safety at compile time
if UnsafeFFI.__info__(:attributes)[:unsafe] do
  IO.warn("Using unsafe module UnsafeFFI")
end
```

### Macros for Module Generation

```elixir
defmodule AutoWrap do
  defmacro wrap_unsafe(module_name, ffi_funcs) do
    quote do
      defmodule unquote(module_name) do
        @unsafe true
        @wrapped_ffi unquote(ffi_funcs)
        
        # Generate safe wrappers automatically
        for {func, arity} <- unquote(ffi_funcs) do
          def unquote(func)(args) do
            # Safe wrapper logic
            apply(:erlang, unquote(func), args)
          end
        end
      end
    end
  end
end

# Usage
AutoWrap.wrap_unsafe(SafeSDL, [
  {:SDL_Init, 1},
  {:SDL_Quit, 0}
])
```

**Why It Works:**
- ✅ Module attributes as metadata
- ✅ Compile-time introspection
- ✅ Macro system for code generation
- ✅ Clear module boundaries

---

## Rust: Crates and Unsafe Blocks (Comparison)

### Module System

```rust
mod vector2d {
    pub fn add(v1: (f64, f64), v2: (f64, f64)) -> (f64, f64) {
        (v1.0 + v2.0, v1.1 + v2.1)
    }
}

use vector2d::add;
```

### Safety Model

```rust
// Safe Rust (default)
fn safe_function() {
    let x = vec![1, 2, 3];
    println!("{}", x[0]);  // Bounds-checked
}

// Unsafe Rust (explicit)
fn dangerous_function() {
    unsafe {
        let ptr = 0x12345 as *const i32;
        let val = *ptr;  // Unsafe dereference
    }
}
```

**Why It Works:**
- ✅ `unsafe` keyword required for dangerous operations
- ✅ Compiler enforces safety boundaries
- ✅ Easy to audit: search for `unsafe`
- ❌ **BUT:** Still requires scattered `unsafe {}` blocks

**What Rust Does Better:**
- FFI calls in separate `extern` blocks
- Unsafe traits clearly marked

```rust
extern "C" {
    fn SDL_Init(flags: u32) -> i32;
}

// All calls to SDL_Init require unsafe
fn init_sdl() {
    unsafe {
        SDL_Init(0);
    }
}
```

---

## Current NanoLang Issues vs Other Languages

### Issue 1: No Module Introspection

**Python:**
```python
import math
dir(math)  # ✅ Lists all exports
```

**Go:**
```bash
go doc math  # ✅ Lists all functions
```

**Ruby:**
```ruby
Math.methods  # ✅ Lists all methods
```

**Elixir:**
```elixir
Math.__info__(:functions)  # ✅ Lists all functions
```

**NanoLang (Current):**
```nano
import "modules/math_ext/math_ext.nano"
// ❌ NO WAY to list exports
// ❌ NO WAY to check if function exists
// ❌ NO WAY to get metadata
```

---

### Issue 2: Scattered Unsafe Blocks

**Go:**
```go
import "unsafe"  // ✅ One line, entire module marked

func do_stuff() {
    ptr := unsafe.Pointer(&x)  // No extra blocks
}
```

**Rust:**
```rust
unsafe fn entire_function_unsafe() {
    // ✅ Function-level unsafe
    raw_ptr_operation();
    another_unsafe_call();
}
```

**NanoLang (Current):**
```nano
fn render() -> void {
    unsafe { (SDL_Init 0) }      // ❌ Noisy
    unsafe { (SDL_CreateWindow) }  // ❌ Noisy
    unsafe { (SDL_Present) }       // ❌ Noisy
    unsafe { (SDL_Quit) }          // ❌ Noisy
}
```

**What Users Want:**
```nano
unsafe module sdl { /* ... */ }

fn render() -> void {
    (SDL_Init 0)       // ✅ Clean
    (SDL_CreateWindow)  // ✅ Clean
    (SDL_Present)       // ✅ Clean
    (SDL_Quit)          // ✅ Clean
}
```

---

### Issue 3: Module Not First-Class

**Python:**
```python
import math

def use_module(mod):
    if hasattr(mod, 'sqrt'):
        return mod.sqrt(16)

use_module(math)  # ✅ Modules are values
```

**Elixir:**
```elixir
defmodule Caller do
  def call_function(module, func, args) do
    apply(module, func, args)  # ✅ Modules are atoms
  end
end

Caller.call_function(Math, :sqrt, [16])
```

**NanoLang (Current):**
```nano
import "modules/math_ext/math_ext.nano" as Math

fn use_module(m: ???) -> int {
    // ❌ No Module type
    // ❌ Can't pass modules as values
    return 0
}
```

---

### Issue 4: No Safety Metadata

**Go (via tooling):**
```bash
$ go-safer check .
Warning: package main imports "unsafe"
  Used in: main.go:45
```

**Python (via type hints):**
```python
from typing import cast, Any

# Type checker warns about unsafe casts
x: int = cast(int, some_object)  # mypy warning
```

**Elixir (via attributes):**
```elixir
if MyModule.__info__(:attributes)[:unsafe] do
  IO.warn("Using unsafe module")
end
```

**NanoLang (Current):**
```nano
import "modules/sdl/sdl.nano"

// ❌ No way to check if module is safe
// ❌ No compiler warnings
// ❌ No metadata available
```

---

## Comparison Table

| Feature | Python | Go | Ruby | Elixir | NanoLang (Current) | NanoLang (Proposed) |
|---------|--------|----|----- |--------|-------------------|---------------------|
| **Module Introspection** | ✅ `dir()` | ✅ `go doc` | ✅ `.methods` | ✅ `__info__` | ❌ None | ✅ `__module_info_*` |
| **Module as Value** | ✅ Yes | ⚠️ Via reflection | ✅ Yes | ✅ Yes | ❌ No | ⏳ Future |
| **Safety Annotation** | ❌ No | ✅ `import "unsafe"` | ❌ No | ✅ Custom attrs | ❌ No | ✅ `unsafe module` |
| **Explicit Imports** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| **Namespace Control** | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes | ⚠️ Limited | ✅ Enhanced |
| **Warning System** | ⚠️ Linters | ✅ `go vet` | ⚠️ Linters | ⚠️ Custom | ❌ No | ✅ Compiler flags |
| **Function-level Unsafe** | N/A | ❌ Block-level | N/A | N/A | ❌ Block-level | ✅ Module-level |
| **Auto-documentation** | ✅ pydoc | ✅ godoc | ✅ rdoc | ✅ ExDoc | ❌ No | ✅ Via metadata |
| **Compile-time Metadata** | ❌ Runtime only | ✅ Yes | ❌ Runtime only | ✅ Yes | ❌ No | ✅ Yes |

---

## Key Insights

### What Makes Go's `unsafe` Work

1. **One import line** - `import "unsafe"` makes entire usage obvious
2. **Tooling integration** - `go vet` flags unsafe usage
3. **No ceremony** - Don't need `unsafe {}` for every call
4. **Clear audit trail** - Search for one string

### What Makes Python's Introspection Work

1. **Modules are objects** - Can inspect like any other object
2. **`dir()` everywhere** - Universal introspection protocol
3. **`__dict__` access** - Full namespace visibility
4. **inspect module** - Rich metadata access

### What Makes Elixir's Attributes Work

1. **Compile-time metadata** - `@moduledoc`, `@vsn`, custom attrs
2. **`__info__/1` protocol** - Standard introspection interface
3. **Macro system** - Generate metadata automatically
4. **Clear module boundaries** - `defmodule` wraps everything

---

## Design Goals for NanoLang

Based on this comparison, NanoLang should have:

### 1. Module-Level Safety (Like Go)

```nano
unsafe module sdl {
    extern fn SDL_Init(flags: int) -> int
    // No unsafe blocks needed inside
}
```

### 2. Compile-Time Introspection (Like Elixir)

```nano
let info: ModuleInfo = (__module_info_sdl)
assert (not info.is_safe)
assert info.has_ffi
```

### 3. Warning System (Like Go vet)

```bash
nanoc app.nano --warn-unsafe-imports
nanoc app.nano --warn-ffi
nanoc app.nano --forbid-unsafe
```

### 4. Module Metadata (Like Python + Elixir)

```nano
struct ModuleInfo {
    name: string,
    version: string,
    is_safe: bool,
    has_ffi: bool,
    exported_functions: array<FunctionInfo>
}
```

### 5. Clear Import Syntax (Like All)

```nano
import safe "modules/vector2d/vector2d.nano"
import unsafe "modules/sdl/sdl.nano"
```

---

## Why This Matters

### For Users

- **Less visual noise** - One `unsafe module` vs dozens of `unsafe {}` blocks
- **Clear safety model** - Know what's safe at a glance
- **Better tooling** - Docs, linters, analyzers can query modules
- **Easier auditing** - `--warn-unsafe-imports` shows all risks

### For Language

- **Competitive feature** - Matches Python/Go/Elixir/Rust
- **Foundation for ecosystem** - Module marketplace, certifications
- **Better error messages** - "Function from unsafe module 'sdl'"
- **Metaprogramming support** - Generate code based on module metadata

### For Ecosystem

- **Module ratings** - Safety scores in package registry
- **Automated audits** - Tools can scan for unsafe usage
- **Dependency analysis** - See safety impact of all dependencies
- **Documentation generation** - Auto-generate from module metadata

---

## Next Steps

1. **Read:** `docs/MODULE_SYSTEM_REDESIGN.md` - Full proposal
2. **Decide:** Module-level safety annotations?
3. **Decide:** Module introspection system?
4. **Decide:** Warning flags?
5. **Implement:** Phase 1 (safety annotations) - 1-2 weeks

---

**Status:** 🔴 **Research Complete** - Ready for architectural decision  
**Date:** 2025-01-08  
**Related:** `docs/MODULE_SYSTEM_REDESIGN.md`
