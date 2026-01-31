# Affine Types in NanoLang: A Practical Guide

**Version:** 0.3.0  
**Status:** Production-Ready  
**Difficulty:** Intermediate

---

## What Are Affine Types?

**Affine types** ensure that resources (like file handles, sockets, or database connections) are **used at most once**. This prevents common bugs at compile time:

- ❌ **Use-after-free**: Reading from a closed file
- ❌ **Use-after-close**: Sending data to a closed socket
- ❌ **Double-free**: Closing the same resource twice
- ❌ **Resource leaks**: Forgetting to close a resource

In NanoLang, affine types are implemented through the `resource struct` keyword, which marks a struct as representing a managed resource with compile-time lifecycle tracking.

---

## Why Affine Types?

### The Problem

```c
// C code - runtime bug waiting to happen
FILE *f = fopen("data.txt", "r");
fread(buffer, 1, 100, f);
fclose(f);
fread(buffer, 1, 100, f);  // BUG: Use after close! 💥
```

### The Solution

```nano
# NanoLang - compile-time safety!
resource struct FileHandle { fd: int }

extern fn open_file(path: string) -> FileHandle
extern fn read_file(f: FileHandle) -> string
extern fn close_file(f: FileHandle) -> void

fn example() -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    let data: string = unsafe { (read_file f) }
    unsafe { (close_file f) }
    # let more: string = unsafe { (read_file f) }  # ✗ COMPILE ERROR!
    return 0
}
```

**The compiler prevents the bug** before your code ever runs!

---

## Quick Start

### Step 1: Declare a Resource Struct

Use the `resource` keyword before `struct`:

```nano
resource struct FileHandle {
    fd: int
}

resource struct Socket {
    id: int,
    port: int
}

resource struct DatabaseConnection {
    handle: int
}
```

**Key Rule**: Any struct declared with `resource` becomes subject to affine type checking.

---

### Step 2: Define Resource Operations

Declare external C functions that work with your resource:

```nano
# Functions that CREATE resources
extern fn open_file(path: string) -> FileHandle
extern fn connect_socket(host: string, port: int) -> Socket

# Functions that USE resources (non-consuming)
extern fn read_file(f: FileHandle) -> string
extern fn send_data(s: Socket, data: string) -> int

# Functions that CONSUME resources (take ownership)
extern fn close_file(f: FileHandle) -> void
extern fn close_socket(s: Socket) -> void
```

**Tip**: Any function that takes a resource by value is considered to "consume" it.

---

### Step 3: Use Resources Safely

```nano
fn safe_file_usage() -> string {
    # 1. Create the resource
    let f: FileHandle = unsafe { (open_file "data.txt") }
    
    # 2. Use it multiple times (OK!)
    let chunk1: string = unsafe { (read_file f) }
    let chunk2: string = unsafe { (read_file f) }
    let chunk3: string = unsafe { (read_file f) }
    
    # 3. Consume it exactly once (REQUIRED!)
    unsafe { (close_file f) }
    
    return (str_concat chunk1 (str_concat chunk2 chunk3))
}

shadow safe_file_usage {
    let result: string = (safe_file_usage)
    assert (>= (str_length result) 0)
}
```

---

## Resource Lifecycle

Every resource goes through these states:

```
UNUSED → USED → CONSUMED
  ↓       ↓        ↓
Create  Use    Transfer ownership
```

### State 1: UNUSED

The resource exists but hasn't been used yet.

```nano
let f: FileHandle = unsafe { (open_file "file.txt") }
# State: UNUSED
```

### State 2: USED

The resource has been borrowed (non-consuming operations).

```nano
let data: string = unsafe { (read_file f) }
# State: USED (can still use it more)

let more: string = unsafe { (read_file f) }
# State: USED (still OK)
```

### State 3: CONSUMED

The resource has been moved/transferred (ownership gone).

```nano
unsafe { (close_file f) }
# State: CONSUMED (cannot use anymore)

# ✗ let x: string = unsafe { (read_file f) }  # COMPILE ERROR!
```

---

## Common Patterns

### Pattern 1: Simple Resource Usage

```nano
fn pattern_simple() -> int {
    let r: FileHandle = unsafe { (open_file "data.txt") }
    let data: string = unsafe { (read_file r) }
    unsafe { (close_file r) }
    return (str_length data)
}

shadow pattern_simple {
    assert (>= (pattern_simple) 0)
}
```

---

### Pattern 2: Multiple Resources

```nano
fn pattern_multiple() -> int {
    # Open multiple resources
    let f1: FileHandle = unsafe { (open_file "file1.txt") }
    let f2: FileHandle = unsafe { (open_file "file2.txt") }
    let s: Socket = unsafe { (connect_socket "localhost" 8080) }
    
    # Use them
    let data1: string = unsafe { (read_file f1) }
    let data2: string = unsafe { (read_file f2) }
    let sent: int = unsafe { (send_data s (str_concat data1 data2)) }
    
    # Close all resources
    unsafe { (close_file f1) }
    unsafe { (close_file f2) }
    unsafe { (close_socket s) }
    
    return sent
}

shadow pattern_multiple {
    assert (>= (pattern_multiple) 0)
}
```

---

### Pattern 3: Conditional Resource Handling

```nano
fn pattern_conditional(should_read: bool) -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    
    if should_read {
        let data: string = unsafe { (read_file f) }
        (println data)
    } else {
        (println "Skipping read")
    }
    
    # Must close in ALL branches!
    unsafe { (close_file f) }
    return 0
}

shadow pattern_conditional {
    assert (== (pattern_conditional true) 0)
    assert (== (pattern_conditional false) 0)
}
```

---

### Pattern 4: Resource with Early Return

```nano
fn pattern_early_return(should_abort: bool) -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    
    if should_abort {
        # Must close before early return!
        unsafe { (close_file f) }
        return (- 0 1)
    }
    
    let data: string = unsafe { (read_file f) }
    unsafe { (close_file f) }
    return (str_length data)
}

shadow pattern_early_return {
    assert (== (pattern_early_return true) (- 0 1))
    assert (>= (pattern_early_return false) 0)
}
```

---

### Pattern 5: Resource in Helper Function

```nano
fn helper_process(f: FileHandle) -> string {
    # Can use the resource here
    return unsafe { (read_file f) }
}

shadow helper_process {
    let f: FileHandle = unsafe { (open_file "test.txt") }
    let result: string = (helper_process f)
    unsafe { (close_file f) }
    (println result)
}

fn pattern_helper() -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    let processed: string = (helper_process f)
    # f is still valid here (helper didn't consume it)
    unsafe { (close_file f) }
    return (str_length processed)
}

shadow pattern_helper {
    assert (>= (pattern_helper) 0)
}
```

---

### Pattern 6: Resource with Loops

```nano
fn pattern_loop() -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    
    let mut total: int = 0
    let mut i: int = 0
    while (< i 5) {
        let chunk: string = unsafe { (read_file f) }
        set total (+ total (str_length chunk))
        set i (+ i 1)
    }
    
    unsafe { (close_file f) }
    return total
}

shadow pattern_loop {
    assert (>= (pattern_loop) 0)
}
```

---

### Pattern 7: Resource in Struct

```nano
struct Connection {
    socket: Socket,
    is_active: bool,
    name: string
}

fn pattern_struct() -> int {
    let s: Socket = unsafe { (connect_socket "localhost" 9000) }
    let conn: Connection = Connection {
        socket: s,
        is_active: true,
        name: "main"
    }
    
    if conn.is_active {
        let sent: int = unsafe { (send_data conn.socket "hello") }
        (println "Message sent")
    }
    
    # Must consume the socket from the struct
    unsafe { (close_socket conn.socket) }
    return 0
}

shadow pattern_struct {
    assert (== (pattern_struct) 0)
}
```

---

## Common Errors and Fixes

### Error 1: Use After Consume

```nano
# ✗ WRONG
let f: FileHandle = unsafe { (open_file "data.txt") }
unsafe { (close_file f) }
let data: string = unsafe { (read_file f) }  # ERROR!

# ✓ CORRECT
let f: FileHandle = unsafe { (open_file "data.txt") }
let data: string = unsafe { (read_file f) }  # Use before consume
unsafe { (close_file f) }
```

**Error Message**:
```
Error: Cannot use resource 'f' after it has been consumed
```

---

### Error 2: Double Consume

```nano
# ✗ WRONG
let f: FileHandle = unsafe { (open_file "data.txt") }
unsafe { (close_file f) }
unsafe { (close_file f) }  # ERROR!

# ✓ CORRECT
let f: FileHandle = unsafe { (open_file "data.txt") }
unsafe { (close_file f) }  # Consume exactly once
```

**Error Message**:
```
Error: Cannot consume resource 'f' - already consumed
```

---

### Error 3: Resource Leak

```nano
# ✗ WRONG
fn leaky() -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    let data: string = unsafe { (read_file f) }
    return (str_length data)
    # ERROR: 'f' was not consumed!
}

# ✓ CORRECT
fn safe() -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    let data: string = unsafe { (read_file f) }
    unsafe { (close_file f) }  # Always consume!
    return (str_length data)
}
```

**Error Message**:
```
Error: Resource 'f' was not consumed before end of scope
```

---

### Error 4: Conditional Leak

```nano
# ✗ WRONG
fn conditional_leak(x: int) -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    if (> x 0) {
        unsafe { (close_file f) }
    }
    # ERROR: 'f' not consumed in else branch!
    return 0
}

# ✓ CORRECT
fn conditional_safe(x: int) -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    if (> x 0) {
        (println "Positive")
    } else {
        (println "Non-positive")
    }
    # Consume in ALL paths
    unsafe { (close_file f) }
    return 0
}
```

---

## Best Practices

### ✅ DO: Consume Resources as Late as Possible

```nano
fn good_practice() -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    let data1: string = unsafe { (read_file f) }
    let data2: string = unsafe { (read_file f) }
    let result: int = (+ (str_length data1) (str_length data2))
    unsafe { (close_file f) }  # Close at the end
    return result
}
```

### ✅ DO: Use Multiple Variables for Multiple Resources

```nano
fn good_multiple() -> int {
    let f1: FileHandle = unsafe { (open_file "file1.txt") }
    let f2: FileHandle = unsafe { (open_file "file2.txt") }
    # Use both
    unsafe { (close_file f1) }
    unsafe { (close_file f2) }
    return 0
}
```

### ✅ DO: Close in All Branches

```nano
fn good_branches(flag: bool) -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    if flag {
        let data: string = unsafe { (read_file f) }
        unsafe { (close_file f) }
        return (str_length data)
    } else {
        unsafe { (close_file f) }
        return 0
    }
}
```

### ❌ DON'T: Try to Clone or Copy Resources

Resources cannot be cloned - they represent unique ownership.

```nano
# ✗ This won't work
let f1: FileHandle = unsafe { (open_file "data.txt") }
let f2: FileHandle = f1  # ERROR: Resources cannot be copied!
```

### ❌ DON'T: Store Resources in Arrays

Resources must have explicit, traceable lifetimes.

```nano
# ✗ This won't work
let handles: array<FileHandle> = [...]  # ERROR: No resource arrays!
```

---

## Real-World Examples

### Example 1: Configuration File Reader

```nano
resource struct FileHandle { fd: int }

extern fn open_file(path: string) -> FileHandle
extern fn read_file(f: FileHandle) -> string
extern fn close_file(f: FileHandle) -> void

fn read_config(path: string) -> string {
    let f: FileHandle = unsafe { (open_file path) }
    let content: string = unsafe { (read_file f) }
    unsafe { (close_file f) }
    return content
}

shadow read_config {
    let config: string = (read_config "config.json")
    assert (>= (str_length config) 0)
}
```

### Example 2: Network Client

```nano
resource struct Socket { id: int }

extern fn connect_socket(host: string, port: int) -> Socket
extern fn send_data(s: Socket, data: string) -> int
extern fn receive_data(s: Socket) -> string
extern fn close_socket(s: Socket) -> void

fn send_request(host: string, request: string) -> string {
    let s: Socket = unsafe { (connect_socket host 80) }
    let sent: int = unsafe { (send_data s request) }
    let response: string = unsafe { (receive_data s) }
    unsafe { (close_socket s) }
    return response
}

shadow send_request {
    let resp: string = (send_request "example.com" "GET / HTTP/1.0\r\n\r\n")
    assert (>= (str_length resp) 0)
}
```

---

## Advanced Topics

### Affine vs Linear Types

- **Linear types**: Must be used **exactly once** (no dropping)
- **Affine types**: Must be used **at most once** (can drop without using)

NanoLang uses **affine types** because they're more flexible:
- You can consume a resource without using it first
- You can conditionally use resources
- Early returns are easier to handle

### Relationship to Rust's Ownership

If you know Rust, NanoLang's affine types are similar to Rust's move semantics:

| Rust | NanoLang |
|------|----------|
| `Drop` trait | Resource consumption |
| Move semantics | Affine types |
| `&mut` borrow | Non-consuming use |
| Lifetime `'a` | Implicit (function scope) |

**Key Difference**: NanoLang's affine types are simpler - no lifetime annotations, no complex borrowing rules.

---

## FAQ

**Q: Can I return a resource from a function?**  
A: Yes! The caller becomes responsible for consuming it.

**Q: What happens if I forget to close a resource?**  
A: Compile error! The type checker enforces cleanup.

**Q: Can resources be optional (nullable)?**  
A: Not yet. Consider using a union wrapper:
```nano
union MaybeFile {
    Some { f: FileHandle },
    None { }
}
```

**Q: Do I need affine types for all structs?**  
A: No! Only for types that represent managed resources (files, sockets, etc.).

**Q: Can I use GC with affine types?**  
A: Yes! Regular values are GC'd. Affine types only track explicit resource cleanup.

---

## Summary

✅ **Affine types** prevent resource bugs at compile time  
✅ Use `resource struct` for resources that need cleanup  
✅ **Create → Use → Consume** is the mandatory lifecycle  
✅ Compiler enforces proper cleanup in all code paths  
✅ **Zero runtime overhead** - all checking at compile time  

**Start using affine types today** for safer, more reliable code!

---

For more details, see:
- **MEMORY.md** - Language reference
- **AFFINE_TYPES_DESIGN.md** - Implementation details
- **tests/test_affine_integration.nano** - Comprehensive examples

