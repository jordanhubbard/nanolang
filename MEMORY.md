# MEMORY.md — NanoLang Machine Distillation

> **Purpose:** Complete machine-readable knowledge base for LLMs, coding agents, and new contributors.
> Everything an AI needs to understand, generate, debug, and extend NanoLang — in one file.
> Pair with `spec.json` for formal grammar and `schema/compiler_schema.json` for AST definitions.

---

## 1. PROJECT IDENTITY

**Name:** NanoLang  
**Version:** 0.2.0  
**License:** Apache 2.0  
**Repo:** `github.com/jordanhubbard/nanolang`  
**Tagline:** A minimal, LLM-friendly programming language with mandatory testing, unambiguous syntax, and formally verified semantics.

**Core Differentiators:**
- Transpiles to C for native performance
- Custom VM backend (NanoISA) with process-isolated FFI
- Core semantics mechanically verified in Coq (zero axioms)
- Self-hosting compiler (bootstraps itself)
- Mandatory shadow tests for every function
- Dual notation (prefix and infix operators)
- Designed specifically for LLM code generation

---

## 2. ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────────────────┐
│                        NanoLang Source (.nano)                       │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
              ┌────────────┴───────────┐
              ▼                        ▼
   ┌────────────────────┐   ┌──────────────────────┐
   │  C Transpile Path  │   │    NanoISA VM Path    │
   │   (bin/nanoc)      │   │   (bin/nano_virt)     │
   └────────┬───────────┘   └──────────┬───────────┘
            │                          │
  Lex → Parse → TypeCheck → Transpile  │  Lex → Parse → TypeCheck → Codegen
            │                          │
            ▼                          ▼
     Generated .c file           .nvm bytecode
            │                          │
            ▼                     ┌────┴────┐
     cc → native binary           │         │
                              nano_vm    native binary
                              (executor)  (embeds VM)
                                  │
                              nano_cop (isolated FFI)
                              nano_vmd (daemon mode)
```

### Compilation Pipeline (Both Paths)

```
Source .nano → Lexer (tokenize) → Parser (AST) → TypeChecker (validate) → Backend
```

**Backend A — C Transpiler:** AST → C source → `cc` → native binary  
**Backend B — NanoISA Codegen:** AST → bytecode → .nvm file or embedded in native binary

---

## 3. BUILD SYSTEM

### Prerequisites
- C99 compiler (`cc`)
- GNU Make (`make` on Linux/macOS, `gmake` on BSD)
- Python 3 (for schema generation bootstrap)
- Optional: Rocq Prover >= 9.0 (for formal proofs), SDL2 (for graphics examples)

### Key Make Targets

| Target | Description |
|--------|-------------|
| `make build` | 3-stage bootstrap: compiles C compiler, self-hosted components, validates |
| `make vm` | Build VM backend: `nano_virt`, `nano_vm`, `nano_cop`, `nano_vmd` |
| `make test` | Full test suite (units + integration + language tests) |
| `make test-quick` | Language tests only (fastest) |
| `make test-vm` | Run test suite through NanoVM backend |
| `make examples` | Build all 150+ example programs |
| `make bootstrap` | Full GCC-style bootstrap (Stage 0 → 1 → 2 → 3) |
| `make clean` | Remove all build artifacts |
| `make schema` | Regenerate compiler schema from `schema/compiler_schema.json` |
| `make modules-index` | Generate module index from manifests |
| `make shadow-check` | Verify shadow tests on changed .nano files |
| `make benchmark` | Run performance benchmarks |
| `make bootstrap-profile` | Build profiled compiler components for self-analysis |

### Build Sentinels

The build system uses sentinel files to track progress and avoid rebuilds:

| Sentinel | Stage |
|----------|-------|
| `.stage1.built` | C reference compiler built |
| `.stage2.built` | Self-hosted components compiled |
| `.stage3.built` | Bootstrap validated |
| `.bootstrap0.built` | Bootstrap Stage 0 (C compiler) |
| `.bootstrap1.built` | Bootstrap Stage 1 (self-hosted nanoc_v06) |
| `.bootstrap2.built` | Bootstrap Stage 2 (recompiled by self) |
| `.bootstrap3.built` | Bootstrap Stage 3 (verified, installed) |

### Key Binaries

| Binary | Location | Description |
|--------|----------|-------------|
| `nanoc` | `bin/nanoc` | Compiler (symlink to `nanoc_c` or `nanoc_stage2`) |
| `nanoc_c` | `bin/nanoc_c` | C reference compiler |
| `nanoc_stage1` | `bin/nanoc_stage1` | Self-hosted compiler (compiled by Stage 0) |
| `nanoc_stage2` | `bin/nanoc_stage2` | Self-hosted compiler (compiled by Stage 1) |
| `nano_virt` | `bin/nano_virt` | NanoISA compiler: .nano → .nvm or native |
| `nano_vm` | `bin/nano_vm` | NanoVM executor: runs .nvm files |
| `nano_cop` | `bin/nano_cop` | Co-process: isolated FFI execution |
| `nano_vmd` | `bin/nano_vmd` | VM daemon: persistent VM process |
| `nanoc-ffi` | `bin/nanoc-ffi` | FFI binding generator |

---

## 4. BOOTSTRAP PROCESS (GCC-STYLE)

NanoLang achieves TRUE self-hosting through a classic 3-stage bootstrap:

### Stage 0 — C Reference Compiler
- **Input:** C source files in `src/`
- **Output:** `bin/nanoc_c` (native binary)
- **Method:** `cc` compiles `src/*.c` → `bin/nanoc_c`
- **This is the "seed" compiler.**

### Stage 1 — Self-Hosted Compiler (First Generation)
- **Input:** `src_nano/nanoc_v06.nano` (the compiler written in NanoLang)
- **Output:** `bin/nanoc_stage1`
- **Method:** `bin/nanoc_c` compiles `nanoc_v06.nano` → `bin/nanoc_stage1`
- **Smoke test:** Must compile and run `examples/language/nl_hello.nano`

### Stage 2 — Recompiled Compiler (Second Generation)
- **Input:** Same `src_nano/nanoc_v06.nano`
- **Output:** `bin/nanoc_stage2`
- **Method:** `bin/nanoc_stage1` compiles `nanoc_v06.nano` → `bin/nanoc_stage2`
- **Smoke test:** Must compile and run `examples/language/nl_hello.nano`

### Stage 3 — Verification
- Compare `nanoc_stage1` and `nanoc_stage2`
- If identical: reproducible build proven (TRUE SELF-HOSTING)
- If different: still validates both work correctly
- Installs `nanoc_stage2` as `bin/nanoc`

### Component Testing (3-Stage Build)

Separate from the bootstrap, the build system also validates self-hosted components:

- **Stage 1:** Build C reference compiler
- **Stage 2:** Compile self-hosted parser, typechecker, transpiler (`src_nano/*_driver.nano`)
- **Stage 3:** Run each compiled component (execute its shadow tests)

---

## 5. LANGUAGE SYNTAX & SEMANTICS

### 5.1 Core Design Principles

1. **One canonical form per construct** — no ambiguity for LLMs
2. **Explicit types always** — no type inference, no implicit conversions
3. **Immutable by default** — `mut` keyword required for mutability
4. **Mandatory shadow tests** — every function must have tests
5. **Dual notation for operators** — both prefix `(+ a b)` and infix `a + b`
6. **Prefix notation for function calls** — always `(f x y)`, never `f(x, y)`
7. **Equal precedence** — all infix operators have equal precedence, left-to-right

### 5.2 Lexical Structure

**Comments:**
```nano
# Single-line comment
/* Multi-line comment */
```

**Keywords:**
```
fn let mut set if else while for in return assert shadow
extern int float bool string void true false print
and or not array struct enum union match cond
import module from as unsafe resource pub opaque
break continue range requires ensures
```

**Tokens (from schema/compiler_schema.json):**
78 token types including: `TOKEN_EOF`, `TOKEN_NUMBER`, `TOKEN_FLOAT`, `TOKEN_STRING`,
`TOKEN_IDENTIFIER`, `TOKEN_TRUE`, `TOKEN_FALSE`, `TOKEN_LPAREN`, `TOKEN_RPAREN`,
`TOKEN_LBRACE`, `TOKEN_RBRACE`, `TOKEN_LBRACKET`, `TOKEN_RBRACKET`, `TOKEN_COMMA`,
`TOKEN_COLON`, `TOKEN_ARROW`, `TOKEN_DOT`, `TOKEN_PLUS`, `TOKEN_MINUS`, `TOKEN_STAR`,
`TOKEN_SLASH`, `TOKEN_PERCENT`, `TOKEN_EQ`, `TOKEN_NE`, `TOKEN_LT`, `TOKEN_LE`,
`TOKEN_GT`, `TOKEN_GE`, `TOKEN_AND`, `TOKEN_OR`, `TOKEN_NOT`, `TOKEN_UNSAFE`,
`TOKEN_RESOURCE`, `TOKEN_MODULE`, `TOKEN_PUB`, `TOKEN_FROM`, `TOKEN_USE`,
`TOKEN_DOUBLE_COLON`, `TOKEN_REQUIRES`, `TOKEN_ENSURES`, `TOKEN_BREAK`,
`TOKEN_CONTINUE`, `TOKEN_COND`

### 5.3 Types

**Primitive Types:**

| Type | C Mapping | Size | Description |
|------|-----------|------|-------------|
| `int` | `int64_t` | 8 bytes | 64-bit signed integer |
| `float` | `double` | 8 bytes | 64-bit IEEE 754 |
| `bool` | `bool` | 1 byte | `true` or `false` |
| `string` | `char*` | 8 bytes (pointer) | UTF-8 text |
| `u8` | `uint8_t` | 1 byte | Unsigned byte |
| `bstring` | `nl_string_t*` | 8 bytes (pointer) | Binary string with length |
| `void` | `void` | 0 bytes | Absence of value |

**Composite Types:**

| Type | Syntax | Description |
|------|--------|-------------|
| `struct` | `struct Name { field: type, ... }` | Product type with named fields |
| `enum` | `enum Name { Variant = N, ... }` | Named integer constants |
| `union` | `union Name { Variant { field: type }, ... }` | Tagged union / sum type |
| `array<T>` | `array<int>`, `[1, 2, 3]` | Dynamic, GC-managed array |
| `List<T>` | `List<int>`, `List<Point>` | Generic list (monomorphized) |
| `HashMap<K,V>` | `HashMap<string, int>` | Generic hash map |
| `(T1, T2, ...)` | `(int, string)` | Tuple |
| `fn(T1, T2) -> R` | `fn(int, int) -> int` | First-class function type |
| `opaque` | `opaque type GLFWwindow` | Opaque C pointer |
| `resource struct` | `resource struct FileHandle { fd: int }` | Affine type (use-at-most-once) |

**Generic Types — Monomorphization:**

NanoLang uses monomorphized generics. `List<int>` becomes `List_int` at compile time.

```nano
# Generic usage
let nums: List<int> = (List_int_new)
(List_int_push nums 42)
let val: int = (List_int_get nums 0)
let len: int = (List_int_length nums)

# Generic Result type
let r: Result<int, string> = Result.Ok { value: 42 }
```

**Generic Union Types:**
```nano
union Result<T, E> {
    Ok { value: T },
    Err { error: E }
}

union Option<T> {
    Some { value: T },
    None {}
}
```

**TypeKind enum (internal representation):**
```
TYPE_INT, TYPE_FLOAT, TYPE_BOOL, TYPE_STRING, TYPE_VOID,
TYPE_ARRAY, TYPE_STRUCT, TYPE_ENUM, TYPE_UNION, TYPE_GENERIC,
TYPE_LIST_INT, TYPE_LIST_STRING, TYPE_LIST_TOKEN, TYPE_LIST_GENERIC,
TYPE_HASHMAP, TYPE_FUNCTION, TYPE_TUPLE, TYPE_OPAQUE, TYPE_UNKNOWN
```

### 5.4 Variables

```nano
let x: int = 42                    # Immutable (default)
let mut counter: int = 0           # Mutable
set counter (+ counter 1)          # Assignment (mut only)
set counter counter + 1            # Infix also works
```

### 5.5 Functions

```nano
fn name(param1: type1, param2: type2) -> return_type {
    return value
}

# External C functions (FFI)
extern fn function_name(param: type) -> return_type

# Public external (for modules)
pub extern fn module_function(param: type) -> return_type
```

**Shadow Tests (MANDATORY):**
```nano
shadow name {
    assert (== (name arg1 arg2) expected)
    assert (condition)
}
```

Every function MUST have a shadow test. Only exception: `extern` functions.
Shadow tests run when the compiled binary executes.

### 5.6 Control Flow

```nano
# If-else (else branch is required)
if condition {
    # then
} else {
    # else
}

# Cond expression (preferred for multi-branch)
(cond
    (condition1 value1)
    (condition2 value2)
    (else default_value))

# While loop
while condition { body }

# For loop (range-based)
for i in (range 0 10) { body }

# Break and continue
while true {
    if done { break }
    if skip { continue }
}
```

### 5.7 Operators

**All binary operators work in both prefix and infix notation.**
**All infix operators have EQUAL precedence, evaluated LEFT-TO-RIGHT.**

```nano
# Arithmetic
(+ a b)    a + b       # Addition (also string concatenation)
(- a b)    a - b       # Subtraction
(* a b)    a * b       # Multiplication
(/ a b)    a / b       # Division
(% a b)    a % b       # Modulo

# Comparison
(== a b)   a == b      # Equal
(!= a b)   a != b      # Not equal
(< a b)    a < b       # Less than
(<= a b)   a <= b      # Less or equal
(> a b)    a > b       # Greater than
(>= a b)   a >= b      # Greater or equal

# Logical
(and p q)  p and q     # Logical AND (short-circuit)
(or p q)   p or q      # Logical OR (short-circuit)
(not p)    not p       # Logical NOT

# Unary
-x                     # Negation
not flag               # Boolean negation
```

**CRITICAL:** No PEMDAS. Use parentheses: `a * (b + c)` not `a * b + c`.

### 5.8 Data Structures

**Structs:**
```nano
struct Point { x: int, y: int }
let p: Point = Point { x: 10, y: 20 }
let x_val: int = p.x               # Field access
```

**Enums:**
```nano
enum Status { Idle = 0, Running = 1, Done = 2 }
let s: int = Status.Running        # Enums are integers
```

**Unions (Tagged Unions):**
```nano
union Shape {
    Circle { radius: float },
    Rectangle { width: float, height: float }
}
let s: Shape = Shape.Circle { radius: 5.0 }
```

**Pattern Matching:**
```nano
match shape {
    Circle(c) => { return (* 3.14159 (* c.radius c.radius)) }
    Rectangle(r) => { return (* r.width r.height) }
}
```

**Arrays (PURELY FUNCTIONAL semantics — all operations return new arrays):**
```nano
let arr: array<int> = [1, 2, 3, 4]
let empty: array<int> = []
let first: int = (at arr 0)
let len: int = (array_length arr)
let mut nums: array<int> = []
set nums (array_push nums 42)           # Returns new array
set nums (array_remove_at nums 0)       # Returns new array
let val: int = (array_pop nums)          # Returns removed element
set nums (array_set nums 0 99)           # Returns new array
```

**Tuples:**
```nano
let coord: (int, int) = (10, 20)
let x: int = coord.0
let y: int = coord.1
```

**HashMaps:**
```nano
let map: HashMap<string, int> = (HashMap_new)
(HashMap_set map "key" 42)
let val: int = (HashMap_get map "key")
let has: bool = (HashMap_has map "key")
```

### 5.9 Module System

```nano
# Import a safe module
module "modules/vector2d/vector2d.nano"

# Import unsafe (FFI) module — no unsafe blocks needed in code
unsafe module "modules/sdl/sdl.nano"

# Import with alias
module "modules/math_helper.nano" as Math
let result: int = (Math.add 2 3)

# Import specific functions
from "modules/std/json/json.nano" import parse, get_string

# Legacy syntax (still supported)
import "modules/old_module.nano"
```

**Module-level `unsafe`:** Declaring `unsafe module` at import means all FFI calls from that module work without individual `unsafe {}` blocks.

### 5.10 Unsafe Blocks

```nano
# Required for extern function calls (unless module declared unsafe)
unsafe {
    set result (some_extern_function arg)
}
```

### 5.11 First-Class Functions

```nano
fn apply(f: fn(int) -> int, x: int) -> int {
    return (f x)
}

fn double(x: int) -> int { return (* x 2) }

shadow apply {
    assert (== (apply double 5) 10)
}

# Function variables
let op: fn(int, int) -> int = add
let result: int = (op 5 3)
```

### 5.12 Affine Types (Resource Safety)

```nano
resource struct FileHandle { fd: int }

fn safe_usage() -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    let data: string = unsafe { (read_file f) }
    unsafe { (close_file f) }       # Consumed — can't use f again
    return 0
}
```

States: `UNUSED` → `USED` → `CONSUMED`. Compile-time enforcement.

### 5.13 Checked Arithmetic

```nano
module "modules/stdlib/checked_math.nano"

let result: Result<int, string> = (checked_add 1000000 2000000)
match result {
    Ok(v) => { (println v.value) },
    Err(e) => { (println e.error) }
}
```

Functions: `checked_add`, `checked_sub`, `checked_mul`, `checked_div`, `checked_mod`.

### 5.14 Contracts (requires/ensures)

```nano
fn sqrt_safe(x: float) -> float
    requires (>= x 0.0)
    ensures (>= result 0.0)
{
    return (sqrt x)
}
```

---

## 6. STANDARD LIBRARY

### 6.1 Core I/O
| Function | Signature | Description |
|----------|-----------|-------------|
| `print` | `(value: any) -> void` | Print without newline |
| `println` | `(value: any) -> void` | Print with newline |
| `assert` | `(condition: bool) -> void` | Runtime assertion |

### 6.2 Math
| Function | Signature |
|----------|-----------|
| `abs` | `(x: int or float) -> int or float` |
| `min` | `(a, b) -> same type` |
| `max` | `(a, b) -> same type` |
| `sqrt` | `(x) -> float` |
| `pow` | `(base, exp) -> float` |
| `floor` | `(x) -> float` |
| `ceil` | `(x) -> float` |
| `round` | `(x) -> float` |
| `sin` | `(x) -> float` (radians) |
| `cos` | `(x) -> float` |
| `tan` | `(x) -> float` |
| `asin` | `(x) -> float` |
| `acos` | `(x) -> float` |
| `atan` | `(x) -> float` |

### 6.3 String Operations
| Function | Description |
|----------|-------------|
| `str_length(s)` | Length in bytes |
| `str_concat(s1, s2)` | Concatenate (or use `+`) |
| `str_substring(s, start, len)` | Extract substring |
| `str_contains(s, substr)` | Substring check |
| `str_index_of(s, substr)` | Find position (-1 if not found) |
| `str_replace(s, old, new)` | Replace occurrences |
| `str_split(s, delim)` | Split to `array<string>` |
| `str_trim(s)` | Trim whitespace |
| `str_to_upper(s)` | Uppercase |
| `str_to_lower(s)` | Lowercase |
| `str_starts_with(s, prefix)` | Prefix check |
| `str_ends_with(s, suffix)` | Suffix check |
| `str_char_at(s, idx)` | Character at index |
| `int_to_string(n)` | Int to string |
| `float_to_string(f)` | Float to string |
| `string_to_int(s)` | Parse integer |

### 6.4 Array Operations
| Function | Description |
|----------|-------------|
| `at(arr, idx)` | Get element (bounds-checked) |
| `array_length(arr)` | Array length |
| `array_push(arr, elem)` | Append (returns new array) |
| `array_pop(arr)` | Remove and return last |
| `array_set(arr, idx, val)` | Set element (returns new array) |
| `array_remove_at(arr, idx)` | Remove at index (returns new array) |
| `array_slice(arr, start, end)` | Slice |
| `range(start, end)` | Range for for-loops |

### 6.5 Generic List Operations

Replace `T` with the concrete type: `List_int_new`, `List_string_push`, `List_Point_get`.

| Function | Description |
|----------|-------------|
| `List_T_new()` | Create new list |
| `List_T_push(list, val)` | Append element |
| `List_T_get(list, idx)` | Get element |
| `List_T_length(list)` | List length |

### 6.6 System / OS
| Function | Description |
|----------|-------------|
| `nl_exec_shell(cmd)` | Execute shell command |
| `file_read(path)` | Read file to string |
| `file_write(path, content)` | Write string to file |
| `file_exists(path)` | Check file existence |
| `get_argc()` / `get_argv(i)` | CLI argument access |
| `nl_os_getpid()` | Process ID |
| `nl_os_getenv(key)` | Environment variable |

### 6.7 Stdlib Modules (in `stdlib/`)

| Module | Purpose |
|--------|---------|
| `stdlib/log.nano` | Structured logging (6 levels: TRACE to FATAL) |
| `stdlib/coverage.nano` | Runtime code coverage tracking |
| `stdlib/timing.nano` | Microsecond-precision benchmarking |
| `stdlib/process.nano` | Process management |
| `stdlib/regex.nano` | Regular expressions |
| `stdlib/ast.nano` | AST manipulation utilities |
| `stdlib/beads.nano` | Issue tracking integration |
| `stdlib/StringBuilder.nano` | Efficient string building |

---

## 7. NANOISA — VIRTUAL MACHINE INSTRUCTION SET

### 7.1 Architecture Model

- **Stack machine** with 5 virtual registers: SP, FP, IP, R0 (accumulator), R1 (scratch)
- **Runtime-typed values:** 16 bytes each (1-byte tag + 15-byte payload)
- **Variable-length encoding:** 1-byte opcode + 0-4 operands
- **Little-endian** byte order
- **178 opcodes** — RISC/CISC hybrid

### 7.2 Value Tags

| Tag | Code | Description |
|-----|------|-------------|
| `TAG_VOID` | 0x00 | No value |
| `TAG_INT` | 0x01 | 64-bit signed integer |
| `TAG_U8` | 0x02 | Unsigned byte |
| `TAG_FLOAT` | 0x03 | 64-bit IEEE 754 double |
| `TAG_BOOL` | 0x04 | true/false |
| `TAG_STRING` | 0x05 | Heap-allocated, GC-managed, immutable |
| `TAG_BSTRING` | 0x06 | Binary string with length |
| `TAG_ARRAY` | 0x07 | Dynamic array, GC-managed |
| `TAG_STRUCT` | 0x08 | Named struct with fields |
| `TAG_ENUM` | 0x09 | Integer variant index |
| `TAG_UNION` | 0x0A | Tagged union |
| `TAG_FUNCTION` | 0x0B | Function table index + optional closure env |
| `TAG_TUPLE` | 0x0C | Fixed-size heterogeneous container |
| `TAG_HASHMAP` | 0x0D | Key-value map |
| `TAG_OPAQUE` | 0x0E | RPC proxy ID (handle to co-process object) |

### 7.3 Opcode Groups

| Range | Group | Key Opcodes |
|-------|-------|-------------|
| 0x00-0x0F | Stack & Constants | `NOP`, `PUSH_I64`, `PUSH_F64`, `PUSH_BOOL`, `PUSH_STR`, `PUSH_VOID`, `DUP`, `POP`, `SWAP`, `ROT3` |
| 0x10-0x1F | Variable Access | `LOAD_LOCAL`, `STORE_LOCAL`, `LOAD_GLOBAL`, `STORE_GLOBAL`, `LOAD_UPVALUE`, `STORE_UPVALUE` |
| 0x20-0x27 | Arithmetic | `ADD`, `SUB`, `MUL`, `DIV`, `MOD`, `NEG` |
| 0x28-0x2F | Comparison | `EQ`, `NE`, `LT`, `LE`, `GT`, `GE` |
| 0x30-0x37 | Logic | `AND`, `OR`, `NOT` |
| 0x38-0x3F | Control Flow | `JMP`, `JMP_TRUE`, `JMP_FALSE`, `CALL`, `CALL_INDIRECT`, `CALL_EXTERN`, `CALL_MODULE`, `RET` |
| 0x40-0x4F | String Ops | `STR_LEN`, `STR_CONCAT`, `STR_SUBSTR`, `STR_CONTAINS`, `STR_EQ`, `STR_CHAR_AT`, `STR_FROM_INT`, `STR_FROM_FLOAT` |
| 0x50-0x5F | Array Ops | `ARR_NEW`, `ARR_PUSH`, `ARR_POP`, `ARR_GET`, `ARR_SET`, `ARR_LEN`, `ARR_SLICE`, `ARR_REMOVE`, `ARR_LITERAL` |
| 0x60-0x67 | Struct Ops | `STRUCT_NEW`, `STRUCT_GET`, `STRUCT_SET`, `STRUCT_LITERAL` |
| 0x68-0x6F | Union/Enum | `UNION_CONSTRUCT`, `UNION_TAG`, `UNION_FIELD`, `MATCH_TAG`, `ENUM_VAL` |
| 0x70-0x77 | Tuple Ops | `TUPLE_NEW`, `TUPLE_GET` |
| 0x78-0x7F | HashMap Ops | `HM_NEW`, `HM_GET`, `HM_SET`, `HM_HAS`, `HM_DELETE`, `HM_KEYS`, `HM_VALUES`, `HM_LEN` |
| 0x80-0x87 | GC/Memory | `GC_RETAIN`, `GC_RELEASE`, `GC_SCOPE_ENTER`, `GC_SCOPE_EXIT` |
| 0x88-0x8F | Type Casts | `CAST_INT`, `CAST_FLOAT`, `CAST_BOOL`, `CAST_STRING`, `TYPE_CHECK` |
| 0x90-0x97 | Closures | `CLOSURE_NEW`, `CLOSURE_CALL` |
| 0xA0-0xAF | I/O & Debug | `PRINT`, `ASSERT`, `DEBUG_LINE`, `HALT` |
| 0xB0-0xBF | Opaque Proxy | `OPAQUE_NULL`, `OPAQUE_VALID` |

### 7.4 Trap Architecture

The VM separates pure computation from side effects:

**Pure core** (`vm_core_execute`) handles 83+ opcodes (arithmetic, logic, data structures, control flow).

When the core encounters a side effect, it returns a **trap descriptor**:

| Trap | Trigger | Handler |
|------|---------|---------|
| `TRAP_EXTERN_CALL` | `OP_CALL_EXTERN` | Route to co-process FFI |
| `TRAP_PRINT` | `OP_PRINT` | Write to stdout |
| `TRAP_ASSERT` | `OP_ASSERT` | Check boolean, abort if false |
| `TRAP_HALT` | `OP_HALT` | Stop execution |
| `TRAP_ERROR` | Runtime error | Report and terminate |

The **harness** (`vm_execute`) dispatches traps and resumes the core.
This enables potential FPGA acceleration of the pure-compute core.

### 7.5 .nvm Binary Format

**Header (32 bytes):**
```
[magic: "NVM\x01" (4B)] [version (4B)] [flags (4B)] [entry_point (4B)]
[section_count (4B)] [string_pool_offset (4B)] [string_pool_length (4B)] [CRC32 (4B)]
```

**Sections:** CODE (0x0001), STRINGS (0x0002), FUNCTIONS (0x0003), STRUCTS (0x0004), ENUMS (0x0005), UNIONS (0x0006), GLOBALS (0x0007), IMPORTS (0x0008), DEBUG (0x0009), METADATA (0x000A), MODULE_REFS (0x000B)

### 7.6 Memory Management

Reference-counted GC with scope-based auto-release:
- `OP_GC_RETAIN` / `OP_GC_RELEASE` — Manual refcount
- `OP_GC_SCOPE_ENTER` / `OP_GC_SCOPE_EXIT` — Automatic release on scope exit
- Compiler inserts scope markers for let-bindings

---

## 8. CO-PROCESS FFI (COP MODEL)

The COP (Co-Process) model isolates FFI calls in a separate process for safety.

### 8.1 Architecture

```
┌─────────────┐    pipes    ┌─────────────┐
│   nano_vm   │ <---------> │  nano_cop   │
│ (pure core) │             │ (FFI calls) │
└─────────────┘             └─────────────┘
```

### 8.2 Protocol

**Wire format:** 8-byte header: `[version (1B)] [msg_type (1B)] [reserved (2B)] [payload_len (4B)]`

**VM to Co-Process:**
- `COP_MSG_INIT` (0x01) — Send import table
- `COP_MSG_FFI_REQ` (0x02) — Call external function
- `COP_MSG_SHUTDOWN` (0x03) — Terminate

**Co-Process to VM:**
- `COP_MSG_FFI_RESULT` (0x10) — Return value
- `COP_MSG_FFI_ERROR` (0x11) — Error string
- `COP_MSG_READY` (0x12) — Init complete

### 8.3 Value Serialization

| Type | Encoding |
|------|----------|
| INT | i64 (8B, little-endian) |
| FLOAT | f64 (8B, IEEE 754) |
| BOOL | u8 (0 or 1) |
| STRING | length (u32) + UTF-8 data |
| ARRAY | elem_type (u8) + count (u32) + elements |
| OPAQUE | i64 proxy ID |
| VOID | 0 bytes |

### 8.4 Lifecycle

1. VM forks `nano_cop` (connected via pipes)
2. VM sends `COP_MSG_INIT` with import table
3. Co-process loads FFI modules, sends `COP_MSG_READY`
4. For each FFI call: VM sends request, co-process returns result
5. VM sends `COP_MSG_SHUTDOWN` on exit

**Key benefit:** If the co-process crashes, the VM detects the broken pipe and recovers gracefully. FFI crashes are fully isolated from VM execution.

### 8.5 VM Daemon (nano_vmd)

Optional persistent VM process listening on Unix domain socket (`/tmp/nanolang_vm_<uid>.sock`).
Accepts .nvm blobs from clients. Reduces startup latency for repeated execution.

---

## 9. FORMAL VERIFICATION (NANOCORE)

### 9.1 Overview

NanoCore is the formally verified subset of NanoLang, mechanized in the Rocq Prover (Coq).

- **~6,170 lines of Coq**
- **193 theorems/lemmas**
- **0 axioms** (fully axiom-free)
- **0 Admitted** proofs

### 9.2 What's Proved

| Theorem | Statement |
|---------|-----------|
| **Preservation** | Well-typed expressions evaluate to values of the expected type |
| **Progress** | Well-typed closed expressions are either values or can step |
| **Determinism** | Evaluation is a partial function (unique results) |
| **Semantic Equivalence** | Big-step and small-step semantics agree |
| **Evaluator Soundness** | Fuel-based evaluator agrees with relational semantics |

### 9.3 Verified Subset (NanoCore)

Integers, booleans, strings, unit type, arithmetic, comparison, logic, string concatenation/length/equality/indexing, if/then/else, let bindings, mutable variables (set), sequential composition, while loops, lambda/application, array literals/indexing/length/push/update, record literals/field access/field update, recursive functions (fix/letrec), variant types, pattern matching.

### 9.4 File Structure

| File | Lines | Contents |
|------|-------|----------|
| `Syntax.v` | 235 | Types, operators, expressions, values, environments |
| `Semantics.v` | 341 | Big-step operational semantics with store-passing |
| `Typing.v` | 293 | Typing rules, contexts |
| `Soundness.v` | 834 | Preservation theorem |
| `Progress.v` | 745 | Small-step semantics, substitution, progress |
| `Determinism.v` | 89 | Determinism proof |
| `Equivalence.v` | 3,098 | Big-step / small-step equivalence (133 lemmas) |
| `EvalFn.v` | 503 | Computable evaluator with soundness proof |
| `Extract.v` | 32 | OCaml extraction configuration |

### 9.5 Key Design Choices

- Big-step semantics with store-passing for preservation
- Scoped let bindings (pop from output environment)
- Lexically scoped function application
- String-based environments (association lists)
- Short-circuit AND/OR
- Total division (div-by-zero produces 0)
- While loop unrolling in small-step
- Structural record typing
- Recursive closures via `VFixClos`
- Exhaustive, ordered branch coverage for variants

### 9.6 Building Proofs

```bash
cd formal/
make                  # Compile all proofs (requires Rocq Prover >= 9.0)
make extract          # Extract OCaml reference interpreter
make nanocore-ref     # Build reference interpreter binary
```

### 9.7 Differential Testing

```bash
make test-differential    # Compare Coq reference interpreter vs NanoVM
```

---

## 10. SOURCE CODE MAP

### 10.1 C Compiler (Stage 0) — `src/`

| File | LOC (approx) | Purpose |
|------|-----|---------|
| `main.c` | 1,343 | Compiler entry point, CLI, orchestration |
| `lexer.c` | 407 | Tokenization |
| `parser.c` | 4,656 | Recursive descent parser producing AST |
| `typechecker.c` | 6,209 | Type checking and validation |
| `transpiler.c` | 4,412 | AST to C code generation |
| `transpiler_iterative_v3_twopass.c` | 3,273 | Two-pass transpilation (included by transpiler.c) |
| `module.c` | 2,268 | Module loading, linking, namespaces |
| `module_builder.c` | 1,716 | Module compilation, auto-instantiation |
| `module_metadata.c` | 529 | Module metadata extraction |
| `stdlib_runtime.c` | 1,875 | Standard library runtime bindings |
| `eval.c` | — | Expression evaluation |
| `nanolang.h` | 1,000+ | Master header: all types, AST nodes, value types |
| `version.h` | 15 | Version: 0.2.0 |

### 10.2 Runtime — `src/runtime/`

| File | Purpose |
|------|---------|
| `gc.c` | Reference-counting garbage collector |
| `gc_struct.c` | GC-managed struct allocations |
| `dyn_array.c` | Dynamic array implementation |
| `nl_string.c` | String operations |
| `ffi_loader.c` | FFI/C library dynamic loading |
| `cli.c` | CLI argument runtime |
| `regex.c` | Regular expression support |
| `token_helpers.c` | Token manipulation helpers |
| `list_*.c` | Monomorphized list types (50+ files) |

### 10.3 NanoISA — `src/nanoisa/`

| File | LOC | Purpose |
|------|-----|---------|
| `isa.c` | 401 | ISA definition, encode/decode |
| `nvm_format.c` | 618 | .nvm binary format, CRC32 |
| `assembler.c` | 736 | Two-pass text assembler |
| `disassembler.c` | 246 | Binary to text with label reconstruction |
| `verifier.c` | — | Bytecode verification |

### 10.4 NanoVM — `src/nanovm/`

| File | LOC | Purpose |
|------|-----|---------|
| `vm.c` | 1,844 | Core switch-dispatch interpreter |
| `value.c` | 225 | NanoValue constructors, type checking |
| `heap.c` | 595 | Reference-counting GC |
| `vm_builtins.c` | 297 | Runtime built-in functions |
| `vm_ffi.c` | 611 | Fork/pipe/exec FFI lifecycle |
| `cop_protocol.c` | 227 | Co-process wire protocol |
| `cop_main.c` | 212 | `nano_cop` binary entry |
| `vmd_protocol.c` | 150 | Daemon wire protocol |
| `vmd_client.c` | 275 | Daemon client connector |
| `vmd_server.c` | 430 | Daemon server handler |
| `vmd_main.c` | 52 | `nano_vmd` entry |
| `main.c` | 214 | `nano_vm` entry |

### 10.5 NanoVirt (Bytecode Compiler) — `src/nanovirt/`

| File | LOC | Purpose |
|------|-----|---------|
| `codegen.c` | 3,083 | AST to bytecode compiler |
| `wrapper_gen.c` | 574 | Native executable generator |
| `main.c` | 331 | `nano_virt` entry |

### 10.6 Self-Hosted Compiler — `src_nano/`

| File | Purpose |
|------|---------|
| `nanoc_v06.nano` | **Main self-hosted compiler driver** |
| `compiler/lexer.nano` | Lexer implementation |
| `parser.nano` | Parser implementation |
| `typecheck.nano` | Type checker |
| `transpiler.nano` | C code generator |
| `compiler/module_loader.nano` | Module loading/resolution |
| `compiler/diagnostics.nano` | Diagnostic system |
| `compiler/error_messages.nano` | Error message formatting |
| `compiler/ir.nano` | Intermediate representation |
| `compiler/serialize.nano` | AST serialization |
| `compiler/result.nano` | Result/error types |
| `generated/compiler_ast.nano` | Auto-generated AST definitions |
| `generated/compiler_schema.nano` | Auto-generated schema types |
| `generated/compiler_contracts.nano` | Auto-generated contracts |
| `ast_shared.nano` | Shared AST definitions |
| `cli_args.nano` | CLI argument parsing |
| `file_io.nano` | File I/O operations |
| `parser_driver.nano` | Parser test driver |
| `typecheck_driver.nano` | Type checker test driver |
| `transpiler_driver.nano` | Transpiler test driver |

### 10.7 Schema Generation

| File | Purpose |
|------|---------|
| `schema/compiler_schema.json` | Master schema: tokens, parse nodes, types, structs |
| `scripts/gen_compiler_schema.py` | Python schema generator (bootstrap) |
| `scripts/gen_compiler_schema.nano` | NanoLang schema generator |

Generated outputs: `src/generated/compiler_schema.h`, `src_nano/generated/compiler_schema.nano`, `src_nano/generated/compiler_ast.nano`, `src_nano/generated/compiler_contracts.nano`

---

## 11. MODULE ECOSYSTEM (49 modules in `modules/`)

**Graphics & UI:** `sdl/`, `sdl_mixer/`, `sdl_ttf/`, `sdl_image/`, `sdl_helpers/`, `glfw/`, `glew/`, `glut/`, `opengl/`, `ncurses/`, `ui_widgets/`

**System & I/O:** `filesystem/`, `stdio/`, `libc/`, `uv/`, `readline/`, `preferences/`

**Data & Networking:** `curl/`, `http_server/`, `sqlite/`, `openai/`, `github/`

**Audio:** `audio_helpers/`, `audio_viz/`, `pt2_audio/`, `pt2_module/`, `pt2_state/`

**Standard Library:** `std/`, `stdlib/`, `vector2d/`, `math_ext/`, `unicode/`, `event/`, `proptest/`, `nano_tools/`, `nano_highlight/`, `tools/`

**AI & Scripting:** `pybridge/`, `pybridge_matplotlib/`, `bullet/`

### Module Structure
```
modules/<name>/
  module.json        # Manifest (name, version, dependencies, platform)
  <name>.nano        # NanoLang interface (extern fn declarations)
  <name>.c           # C implementation (FFI bridge)
  mvp.nano           # Self-test program
  README.md          # Documentation
```

`modules/index.json` — auto-generated by `make modules-index`.

---

## 12. TESTING

### 12.1 Test Architecture

| Test Type | Location | Runner |
|-----------|----------|--------|
| Language tests | `tests/nl_*.nano`, `tests/test_*.nano` | `tests/run_all_tests.sh` |
| C unit tests | `tests/test_transpiler.c` | `make test-units` |
| NanoISA tests | `tests/nanoisa/test_nanoisa.c` (470 tests) | `make test-nanoisa` |
| NanoVM tests | `tests/nanovm/test_vm.c` (150 tests) | `make test-nanovm` |
| NanoVirt tests | `tests/nanovirt/test_codegen.c` (62 tests) | `make test-nanovirt` |
| Self-host tests | `tests/selfhost/` | `tests/selfhost/run_selfhost_tests.sh` |
| Negative tests | `tests/negative/` | `tests/run_negative_tests.sh` |
| Integration tests | `tests/integration/` | Via `run_all_tests.sh` |
| Regression tests | `tests/regression/` | Via `run_all_tests.sh` |
| Fuzzing tests | `tests/fuzzing/` | Manual |
| Performance tests | `tests/performance/` | `make benchmark` |
| Differential tests | `tests/differential/` | `make test-differential` |
| User guide tests | `tests/user_guide/` | `make test-docs` |

### 12.2 Running Tests

```bash
make test              # Full suite (build + all tests)
make test-quick        # Language tests only (fastest)
make test-vm           # Tests through NanoVM backend
make test-daemon       # Tests through VM daemon
make test-stage1       # Tests with C reference compiler only
make test-bootstrap    # Tests with fully bootstrapped compiler
make test-lang         # Core language tests (nl_* only)
make test-docs         # User guide snippet tests
make test-nanoisa      # NanoISA unit tests
make test-nanovm       # NanoVM unit tests
make test-nanovirt     # NanoVirt codegen tests
make test-differential # Coq reference vs NanoVM
```

### 12.3 CI/CD

GitHub Actions: `.github/workflows/ci.yml`, `.github/workflows/userguide_pages.yml`

---

## 13. EXAMPLES (150+)

### Learning Path

**Level 1 (Beginner):** `nl_hello.nano` → `nl_calculator.nano` → `nl_mutable.nano` → `nl_control_if_while.nano` → `nl_control_for.nano`

**Level 2 (Core):** `nl_functions_basic.nano` → `nl_array_complete.nano` → `nl_struct.nano` → `nl_string_operations.nano` → `nl_factorial.nano`

**Level 3 (Intermediate):** `nl_types_union_construct.nano` → `nl_generics_demo.nano` → `nl_hashmap.nano` → `namespace_demo.nano` → `nl_extern_string.nano`

**Level 4 (Advanced):** Games (Snake, Checkers, Asteroids), Graphics (Boids, Particles, Raytracer), Systems (HTTP server, AI agents)

### Categories

| Category | Directory | Description |
|----------|-----------|-------------|
| Language | `examples/language/` | 50+ core feature demos |
| Advanced | `examples/advanced/` | FFI, modules, performance |
| Games | `examples/games/` | SDL games (Pong, Asteroids, Checkers) |
| Terminal | `examples/terminal/` | NCurses apps (Snake, Life, Matrix) |
| Audio | `examples/audio/` | SDL audio, MOD player |
| Graphics | `examples/graphics/` | Particles, boids, raytracer |
| Network | `examples/network/` | HTTP server, REST API, curl |
| Verified | `examples/verified/` | Formally verified patterns |
| OpenGL | `examples/opengl/` | 3D graphics |
| Playground | `examples/playground/` | Web-based playground server |
| OPL | `examples/opl/` | Custom parser language (showcase) |

---

## 14. COMMON PATTERNS & IDIOMS

### The NanoLang Pattern

```nano
# 1. Types first
struct Point { x: int, y: int }

# 2. Constructor function
fn Point_new(x: int, y: int) -> Point {
    return Point { x: x, y: y }
}

# 3. Shadow test immediately after
shadow Point_new {
    let p: Point = (Point_new 5 10)
    assert (== p.x 5)
    assert (== p.y 10)
}

# 4. Operations on the type
fn Point_add(a: Point, b: Point) -> Point {
    return Point { x: (+ a.x b.x), y: (+ a.y b.y) }
}

shadow Point_add {
    let a: Point = (Point_new 1 2)
    let b: Point = (Point_new 3 4)
    let c: Point = (Point_add a b)
    assert (== c.x 4)
    assert (== c.y 6)
}

# 5. Main entry point
fn main() -> int {
    let p: Point = (Point_new 10 20)
    (println p.x)
    return 0
}

shadow main { assert true }
```

### Error Handling Pattern

```nano
fn divide(a: int, b: int) -> Result<int, string> {
    if (== b 0) {
        return Result.Err { error: "Division by zero" }
    }
    return Result.Ok { value: (/ a b) }
}

shadow divide {
    let r1: Result<int, string> = (divide 10 2)
    match r1 {
        Ok(v) => assert (== v.value 5),
        Err(e) => assert false
    }
}
```

### Naming Conventions

- Functions: `snake_case` — `calculate_distance`, `parse_input`
- Types: `PascalCase` — `Point`, `GameState`, `Result`
- Variables: `snake_case` — `total_count`, `is_valid`
- Generic functions: `Type_operation` — `List_int_push`, `Point_new`
- Constants: expressed as enums — `enum Color { Red = 0, Blue = 1 }`

---

## 15. COMMON ERRORS & DEBUGGING

| Error | Cause | Fix |
|-------|-------|-----|
| `Function 'foo' missing shadow test` | No shadow block | Add `shadow foo { assert ... }` |
| `Type mismatch in let statement` | Wrong type annotation | Check types match exactly |
| `Unexpected token` | C-style function call | Use `(f x y)` not `f(x, y)` |
| `Cannot assign to immutable variable` | Missing `mut` | Declare with `let mut` |
| `If statement requires else branch` | Missing else | Add `} else { ... }` |
| `Undefined variable/function` | Not declared before use | Declare before use |
| `Extern call outside unsafe block` | FFI safety | Wrap in `unsafe { }` or use `unsafe module` |

### Compiler Flags

| Flag | Description |
|------|-------------|
| `-o <path>` | Output binary path |
| `--keep-c` | Keep generated C file |
| `-v` | Verbose output |
| `--llm-diags-json <path>` | Machine-readable diagnostics |
| `-pg` | Enable profiling |
| `--run` | (nano_virt) Execute immediately |
| `--emit-nvm` | (nano_virt) Output .nvm bytecode |
| `--isolate-ffi` | (nano_vm) Route FFI through co-process |
| `--daemon` | (nano_vm) Send to daemon |

---

## 16. KNOWN ISSUES & LIMITATIONS

### Active Known Issues

- **`from ... import` infinite loop (bead nl-hmt, P0):** Self-hosted stage1 compiler infinite loops with `from ... import` + multiple modules. Workaround: use `module`/`import` instead.
- **No closures/lambdas as expressions** — functions are first-class but no anonymous inline closures
- **No generic functions** — only generic types (List<T>, Result<T,E>)
- **No trait/interface system** — polymorphism via monomorphization only
- **No REPL** — interpreter removed; all compilation is ahead-of-time

### Design Limitations (by choice)

- All infix operators have equal precedence (use parens)
- `else` branch required on all `if` statements
- No type inference — all types explicit
- No implicit conversions
- No operator overloading
- No exceptions — use Result unions
- Division by zero produces 0 (matches Coq formalization)

---

## 17. ISSUE TRACKING (BEADS)

NanoLang uses `bd` (beads) for issue tracking, stored in `.beads/`.

```bash
bd onboard             # Get started
bd ready               # Find available work
bd show <id>           # View issue details
bd update <id> --status in_progress   # Claim work
bd close <id>          # Complete work
bd sync                # Sync with git
```

---

## 18. CODE GENERATION CHECKLIST (FOR LLMs)

Before generating NanoLang code, verify:

- [ ] All functions have shadow tests
- [ ] Operators use prefix `(+ a b)` or infix `a + b` (both valid)
- [ ] All variables have explicit types: `let x: int = 5`
- [ ] All if statements have else branches
- [ ] Mutable variables declared with `let mut`
- [ ] Function parameters have type annotations
- [ ] Function return type is specified
- [ ] Struct field access uses dot notation: `point.x`
- [ ] Function calls use prefix: `(function arg1 arg2)`
- [ ] No C-style calls: never `f(x, y)`
- [ ] Range loops: `for i in (range 0 10)`
- [ ] String concat: `(+ str1 str2)` or `str1 + str2`
- [ ] Array ops return new arrays: `set arr (array_push arr val)`
- [ ] FFI calls wrapped in `unsafe {}` or module declared `unsafe module`
- [ ] Use `cond` for multi-branch expressions, `if/else` for statements
- [ ] Use parentheses to control operator grouping (no precedence)
- [ ] `main` function returns `int` and has shadow test
- [ ] Commas make tuples: `(a, b)` is tuple, `(f a b)` is call

---

## 19. PLATFORM SUPPORT

| Platform | Status | Notes |
|----------|--------|-------|
| Ubuntu 22.04+ (x86_64, ARM64) | Fully supported | Primary development platform |
| macOS 14+ (Apple Silicon) | Fully supported | `make` uses GNU make by default |
| FreeBSD | Fully supported | Use `gmake` instead of `make` |
| Windows | WSL2 only | Use WSL2 with Ubuntu |

---

## 20. QUICK REFERENCE CARD

```nano
# --- VARIABLES ------------------------------------------------
let x: int = 42                    # Immutable
let mut y: int = 0                 # Mutable
set y (+ y 1)                      # Assign

# --- FUNCTIONS ------------------------------------------------
fn add(a: int, b: int) -> int {
    return (+ a b)
}
shadow add {
    assert (== (add 2 3) 5)
}

# --- CONTROL FLOW ---------------------------------------------
if (> x 0) { ... } else { ... }
while (< i 10) { set i (+ i 1) }
for i in (range 0 10) { ... }
(cond ((< x 0) "neg") ((== x 0) "zero") (else "pos"))
match val { Ok(v) => { ... } Err(e) => { ... } }

# --- TYPES ----------------------------------------------------
struct Point { x: int, y: int }
enum State { On = 0, Off = 1 }
union Result<T, E> { Ok { value: T }, Err { error: E } }
let arr: array<int> = [1, 2, 3]
let tup: (int, string) = (42, "hello")

# --- MODULES --------------------------------------------------
module "modules/math.nano" as Math
unsafe module "modules/sdl/sdl.nano"
from "modules/lib.nano" import func1, func2

# --- OPERATORS ------------------------------------------------
(+ a b)  a + b   # Arithmetic: + - * / %
(== a b) a == b   # Comparison: == != < <= > >=
(and p q) p and q # Logic: and or not
(f x y)           # Function call (always prefix)
```

---

*Last updated: 2026-02-20. Generated from full codebase analysis.*
*Pair with `spec.json` for formal grammar and `schema/compiler_schema.json` for AST definitions.*
