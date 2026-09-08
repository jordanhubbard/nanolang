# Affine Types: A Practical Guide

**Version:** 5.0 design draft
**Status:** Target semantics; current implementation is partial
**Difficulty:** Intermediate

---

This guide is a proposal for my Phase 20 ownership contract. It is not a
statement that every example works today. My C seed currently parses resource
structs and performs basic identifier-state checks; `src_nano`, path-sensitive
control flow, borrowing, nested ownership, and verified NanoISA metadata remain
5.0 work. The task-backed plan is in `AFFINE_TYPES_DESIGN.md` and
`ROADMAP.md`. Until that plan passes its acceptance matrix, I do not promise
production resource safety.

## What Are My Affine Types?

My 5.0 affine contract will keep ownership of resources, such as file handles,
sockets, and database connections, unique. The completed checker will prevent:

- Use-after-free: Reading from a closed file
- Use-after-close: Sending data to a closed socket
- Double-free: Closing the same resource twice
- Resource leaks: Forgetting to close a resource

I mark managed resources with `resource struct`. The current C-seed prototype
tracks a subset of that lifecycle; the 5.0 work closes the gaps named above.

---

## Why I Use Affine Types

### The Problem

```c
// C code - runtime bug waiting to happen
FILE *f = fopen("data.txt", "r");
fread(buffer, 1, 100, f);
fclose(f);
fread(buffer, 1, 100, f);  // BUG: Use after close! 💥
```

### My Solution

```nano
# NanoLang - compile-time safety!
resource struct FileHandle { fd: int }

extern fn open_file(path: string) -> FileHandle
extern fn read_file(f: &FileHandle) -> string
extern fn close_file(f: FileHandle) -> void

fn example() -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    let data: string = unsafe { (read_file &f) }
    unsafe { (close_file f) }
    # let more: string = unsafe { (read_file &f) }  # COMPILE ERROR
    return 0
}
```

This is the compile-time behavior I require for 5.0.

---

## Quick Start

### Step 1: Declare a Resource Struct

Use my `resource` keyword before `struct`:

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

Any struct I see declared with `resource` becomes subject to my affine type checking.

---

### Step 2: Define Resource Operations

Declare external C functions that work with your resource:

```nano
# Functions that CREATE resources
extern fn open_file(path: string) -> FileHandle
extern fn connect_socket(host: string, port: int) -> Socket

# Functions that BORROW resources (non-consuming, proposed 5.0 syntax)
extern fn read_file(f: &FileHandle) -> string
extern fn send_data(s: &Socket, data: string) -> int

# Functions that CONSUME resources (take ownership)
extern fn close_file(f: FileHandle) -> void
extern fn close_socket(s: Socket) -> void
```

When a function takes a resource by value, I consider ownership transferred.
Repeated reads require the borrowing form. The 5.0 contract task must accept
this syntax and define its aliasing limits before these examples become
normative.

---

### Step 3: Use Resources Safely

```nano
fn safe_file_usage() -> string {
    # 1. Create the resource
    let f: FileHandle = unsafe { (open_file "data.txt") }
    
    # 2. Use it multiple times (OK!)
    let chunk1: string = unsafe { (read_file &f) }
    let chunk2: string = unsafe { (read_file &f) }
    let chunk3: string = unsafe { (read_file &f) }
    
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

I track every resource through these states:

```
UNUSED → USED → CONSUMED
  ↓       ↓        ↓
Create  Use    Transfer ownership
```

### State 1: UNUSED

The resource exists. You have not used it yet.

```nano
let f: FileHandle = unsafe { (open_file "file.txt") }
# State: UNUSED
```

### State 2: USED

You have borrowed the resource through non-consuming operations.

```nano
let data: string = unsafe { (read_file &f) }
# State: USED (can still use it more)

let more: string = unsafe { (read_file &f) }
# State: USED (still OK)
```

### State 3: CONSUMED

You have moved or transferred the resource. Its ownership is gone.

```nano
unsafe { (close_file f) }
# State: CONSUMED (cannot use anymore)

# let x: string = unsafe { (read_file &f) }  # COMPILE ERROR
```

---

## Common Patterns

### Pattern 1: Simple Resource Usage

```nano
fn pattern_simple() -> int {
    let r: FileHandle = unsafe { (open_file "data.txt") }
    let data: string = unsafe { (read_file &r) }
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
    let data1: string = unsafe { (read_file &f1) }
    let data2: string = unsafe { (read_file &f2) }
    let sent: int = unsafe { (send_data &s (str_concat data1 data2)) }
    
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
        let data: string = unsafe { (read_file &f) }
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
    
    let data: string = unsafe { (read_file &f) }
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
fn helper_process(f: &FileHandle) -> string {
    # I borrow the resource here
    return unsafe { (read_file f) }
}

shadow helper_process {
    let f: FileHandle = unsafe { (open_file "test.txt") }
    let result: string = (helper_process &f)
    unsafe { (close_file f) }
    (println result)
}

fn pattern_helper() -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    let processed: string = (helper_process &f)
    # f is still valid here because helper borrowed it
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
        let chunk: string = unsafe { (read_file &f) }
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
        let sent: int = unsafe { (send_data &conn.socket "hello") }
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

## Errors I Detect

### Error 1: Use After Consume

```nano
# ✗ WRONG
let f: FileHandle = unsafe { (open_file "data.txt") }
unsafe { (close_file f) }
let data: string = unsafe { (read_file &f) }  # ERROR

# ✓ CORRECT
let f: FileHandle = unsafe { (open_file "data.txt") }
let data: string = unsafe { (read_file &f) }  # Borrow before consume
unsafe { (close_file f) }
```

My error message:
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

My error message:
```
Error: Cannot consume resource 'f' - already consumed
```

---

### Error 3: Resource Leak

```nano
# ✗ WRONG
fn leaky() -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    let data: string = unsafe { (read_file &f) }
    return (str_length data)
    # ERROR: 'f' was not consumed!
}

# ✓ CORRECT
fn safe() -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    let data: string = unsafe { (read_file &f) }
    unsafe { (close_file f) }  # Always consume!
    return (str_length data)
}
```

My error message:
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

## Things I Recommend

### Consume Resources as Late as Possible

```nano
fn good_practice() -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    let data1: string = unsafe { (read_file &f) }
    let data2: string = unsafe { (read_file &f) }
    let result: int = (+ (str_length data1) (str_length data2))
    unsafe { (close_file f) }  # Close at the end
    return result
}
```

### Use Multiple Variables for Multiple Resources

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

### Close in All Branches

```nano
fn good_branches(flag: bool) -> int {
    let f: FileHandle = unsafe { (open_file "data.txt") }
    if flag {
        let data: string = unsafe { (read_file &f) }
        unsafe { (close_file f) }
        return (str_length data)
    } else {
        unsafe { (close_file f) }
        return 0
    }
}
```

### Do Not Try to Clone or Copy Resources

Resources cannot be cloned. They represent unique ownership.

```nano
# ✗ This won't work
let f1: FileHandle = unsafe { (open_file "data.txt") }
let f2: FileHandle = f1  # ERROR: Resources cannot be copied!
```

### Treat Resource Collections as Unspecified in 5.0

The current checker does not support resource-bearing arrays. Whether I reject
them permanently or define element ownership is an explicit semantic-contract
decision; code must not rely on them before that decision lands.

```nano
# Not accepted by the current contract
let handles: array<FileHandle> = [...]
```

---

## Real-World Examples

### Example 1: Configuration File Reader

```nano
resource struct FileHandle { fd: int }

extern fn open_file(path: string) -> FileHandle
extern fn read_file(f: &FileHandle) -> string
extern fn close_file(f: FileHandle) -> void

fn read_config(path: string) -> string {
    let f: FileHandle = unsafe { (open_file path) }
    let content: string = unsafe { (read_file &f) }
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
extern fn send_data(s: &Socket, data: string) -> int
extern fn receive_data(s: &Socket) -> string
extern fn close_socket(s: Socket) -> void

fn send_request(host: string, request: string) -> string {
    let s: Socket = unsafe { (connect_socket host 80) }
    let sent: int = unsafe { (send_data &s request) }
    let response: string = unsafe { (receive_data &s) }
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

- Linear values must be used exactly once; implicit weakening is forbidden.
- Affine ownership cannot be duplicated and may be transferred at most once.

My proposed resource model combines affine no-copy ownership with a mandatory
ownership-ending action on every reachable exit. An explicit `drop`, if the
contract accepts it, is such an action; silently abandoning a live operating
system handle is not. The contract task will settle the name and exact rule.

### Relationship to Rust's Ownership

If you know Rust, my affine types are similar to Rust's move semantics:

| Rust | NanoLang (Me) |
|------|----------|
| `Drop` trait | Resource consumption |
| Move semantics | Affine types |
| `&` / `&mut` borrow | Proposed non-consuming use |
| Lifetime `'a` | Implicit (function scope) |

My target remains smaller than Rust's ownership system. I do not yet claim a
working borrow checker or stable borrowing syntax.

---

## FAQ

**Can I return a resource from a function?**  
That is the 5.0 target: the caller becomes responsible for consuming it.

**What happens if I forget to close a resource?**  
The 5.0 acceptance matrix requires a compile error on every reachable leaking
exit. The current prototype does not yet prove that guarantee.

**Can resources be optional (nullable)?**  
Not yet. Consider using a union wrapper:
```nano
union MaybeFile {
    Some { f: FileHandle },
    None { }
}
```

**Do I need affine types for all structs?**  
No. Only for types that represent managed resources like files or sockets.

**Can I use GC with affine types?**  
Yes. Regular values are GC'd. Affine types only track explicit resource cleanup.

---

## Summary

- My 5.0 target prevents use-after-transfer, duplicate ownership, double
  consume, and resource leaks at compile time.
- `resource struct` marks values that participate in that ownership contract.
- Borrowing observes; moving or passing by value transfers; cleanup or an
  accepted explicit discard ends ownership.
- I will claim all-path enforcement only after both frontends and the verified
  `.nvm` pipeline pass the same acceptance matrix.
- The design adds no required runtime ownership bookkeeping.

---

For more details, see:
- **MEMORY.md** - My language reference
- **AFFINE_TYPES_DESIGN.md** - My 5.0 contract and task plan
- **tests/test_affine_integration.nano** - Current C-seed examples, not the
  complete 5.0 conformance suite
