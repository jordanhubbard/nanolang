# Affine Types for Resource Safety

**Status**: Implementation in progress  
**Issue**: nanolang-683j  
**Decision**: Affine types for resources + GC for everything else  

## Problem Statement

Prevent use-after-close errors at compile time:

```nano
let file: FileHandle = (open "data.txt")
(close file)
let data: string = (read file)  // Should be compile error, not runtime error!
```

## Design Decision

**NOT** using:
- ❌ Full borrow checker (too complex, months of work × 2 for dual impl)
- ❌ Pure ARC (doesn't solve use-after-close, needs weak refs for cycles)

**USING**:
- ✅ Affine types for resources (use at most once)
- ✅ Keep GC for strings, arrays, structs (99% of code)
- ✅ Compile-time only checking (zero runtime overhead)

## Core Concepts

### Affine Types
A type is **affine** if values can be used **at most once**. After use, the value is "consumed" and cannot be accessed again.

### Resource Types
Types marked as `resource` are affine and represent system resources that must be explicitly released:

```nano
resource struct FileHandle {
    fd: int
}

resource struct Socket {
    sockfd: int
}

resource struct GpuBuffer {
    handle: int
}
```

### Consuming Functions
Functions that take ownership and consume a resource:

```nano
/* This function consumes the FileHandle */
fn close(f: FileHandle) -> void {
    unsafe { (c_close f.fd) }
}

/* After calling close(f), f cannot be used again */
```

## Syntax

### Declaring Resource Types

```nano
resource struct FileHandle {
    fd: int
}
```

### Resource Functions

```nano
/* Returns a resource - ownership transferred to caller */
fn open(path: string) -> FileHandle {
    let fd: int = unsafe { (c_open path 0) }
    return FileHandle { fd: fd }
}

/* Consumes a resource - takes ownership */
fn close(f: FileHandle) -> void {
    unsafe { (c_close f.fd) }
}

/* Borrows a resource - doesn't consume */
fn read(f: &FileHandle, buf: array<int>) -> int {
    return unsafe { (c_read f.fd buf) }
}
```

Note: `&` syntax for borrowing is optional for MVP. Start with move-only semantics.

## Compiler Rules

### Rule 1: Resources Must Be Used Exactly Once (or explicitly dropped)

```nano
fn good() {
    let f: FileHandle = (open "test.txt")
    (close f)  // OK - consumed exactly once
}

fn bad1() {
    let f: FileHandle = (open "test.txt")
    // ERROR: f not consumed (resource leak)
}

fn bad2() {
    let f: FileHandle = (open "test.txt")
    (close f)
    (close f)  // ERROR: f already consumed
}
```

### Rule 2: Resources Cannot Be Copied

```nano
fn bad() {
    let f1: FileHandle = (open "test.txt")
    let f2: FileHandle = f1  // ERROR: Cannot copy resource type
    (close f1)
}
```

### Rule 3: Resources Move on Assignment

```nano
fn good() {
    let f1: FileHandle = (open "test.txt")
    let f2: FileHandle = f1  // OK - f1 moved to f2, f1 no longer accessible
    (close f2)               // OK
    // (close f1)            // ERROR: f1 was moved
}
```

### Rule 4: Resources in Structs

```nano
struct Config {
    name: string       // GC'd - can copy
    file: FileHandle   // Resource - cannot copy
}

fn use_config() {
    let c: Config = Config {
        name: "test",
        file: (open "config.txt")
    }
    // Config itself becomes affine because it contains a resource
    // Must consume c.file before c goes out of scope
    (close c.file)
}
```

## Implementation Plan

### Phase 1: Parser Changes (Week 1)
- [ ] Add `resource` keyword to lexer (TOKEN_RESOURCE)
- [ ] Parse `resource struct` declarations
- [ ] Add `resource` flag to AST_STRUCT node
- [ ] Tests: Parse resource struct declarations

### Phase 2: Type System (Week 2)
- [ ] Add `is_resource` flag to Type and StructInfo
- [ ] Track resource usage in Environment/TypeChecker
- [ ] Implement use-tracking per variable
  - unused, used, consumed states
- [ ] Error on double-use of consumed resources
- [ ] Error on unused resources (resource leak)
- [ ] Tests: Type checking resource usage

### Phase 3: Function Analysis (Week 3)
- [ ] Detect consuming functions (take resource by value)
- [ ] Detect borrowing functions (take resource by reference - optional)
- [ ] Track resource flow through function calls
- [ ] Error on returning consumed resources
- [ ] Tests: Resource flow through functions

### Phase 4: Control Flow (Week 4)
- [ ] Track resource usage in if/else branches
- [ ] Ensure resources consumed on all paths
- [ ] Track resources in while loops
- [ ] Error on potential leaks in conditional code
- [ ] Tests: Conditional resource usage

### Phase 5: Transpiler (Week 5)
- [ ] Transpile resource types as regular structs (no special handling)
- [ ] Emit warnings in generated C for consumed resources
- [ ] Tests: Generated C compiles and runs

### Phase 6: Standard Library (Week 6)
- [ ] Define FileHandle as resource type
- [ ] Define Socket as resource type
- [ ] Update fs module to use resource types
- [ ] Add examples showing resource safety
- [ ] Documentation and examples

## Examples

### File I/O with Resource Safety

```nano
resource struct FileHandle {
    fd: int
}

fn open(path: string) -> FileHandle {
    let fd: int = unsafe { (c_open path 0) }
    return FileHandle { fd: fd }
}

fn close(f: FileHandle) -> void {
    unsafe { (c_close f.fd) }
}

fn read_file(path: string) -> string {
    let f: FileHandle = (open path)
    let mut data: string = ""
    unsafe {
        set data (c_read_all f.fd)
    }
    (close f)  // Must close before returning
    return data
}
```

### Socket with Resource Safety

```nano
resource struct Socket {
    sockfd: int
}

fn connect(host: string, port: int) -> Socket {
    let fd: int = unsafe { (c_connect host port) }
    return Socket { sockfd: fd }
}

fn close_socket(s: Socket) -> void {
    unsafe { (c_close s.sockfd) }
}

fn send_http_request(host: string) -> string {
    let sock: Socket = (connect host 80)
    unsafe { (c_send sock.sockfd "GET / HTTP/1.1\r\n\r\n") }
    let mut response: string = ""
    unsafe {
        set response (c_recv sock.sockfd 4096)
    }
    (close_socket sock)
    return response
}
```

### Error Handling with Resources

```nano
/* Option 1: Manual error handling */
fn safe_read_file(path: string) -> string {
    let f: FileHandle = (open path)
    if (== f.fd -1) {
        /* open failed, but we still have a FileHandle to clean up */
        (close f)
        return ""
    }
    let mut data: string = ""
    unsafe {
        set data (c_read_all f.fd)
    }
    (close f)
    return data
}

/* Option 2: Result type (future work) */
fn safe_read_file_v2(path: string) -> Result<string, string> {
    let f: FileHandle = (open path) or return Err("Failed to open")
    let data: string = unsafe { (c_read_all f.fd) }
    (close f)
    return Ok(data)
}
```

## Future Enhancements

### Borrowing (Phase 7 - Optional)
Allow temporary access without consuming:

```nano
fn read_first_line(f: &FileHandle) -> string {
    /* Borrows f, doesn't consume it */
    return unsafe { (c_read_line f.fd) }
}

fn main() {
    let f: FileHandle = (open "test.txt")
    let line: string = (read_first_line &f)  // Borrow
    (println line)
    (close f)  // Can still close because we only borrowed
}
```

### Explicit Drop
Allow early resource release:

```nano
fn use_file() {
    let f: FileHandle = (open "test.txt")
    let data: string = (read_all f)
    drop f  // Explicitly consume without calling close
}
```

### Resource Pools
For reusable resources:

```nano
resource struct Connection {
    conn: int
}

struct Pool {
    connections: array<Connection>  // Can store resources in collections
}

fn borrow_from_pool(pool: &mut Pool) -> Connection {
    /* Remove and return a connection from pool */
}

fn return_to_pool(pool: &mut Pool, conn: Connection) {
    /* Add connection back to pool */
}
```

## Comparison with Other Approaches

| Approach | Prevents use-after-close | Runtime overhead | Complexity | Handles cycles |
|----------|-------------------------|------------------|------------|----------------|
| **Affine Types** | ✅ Yes | ❌ None | ✅ Low | N/A (uses GC) |
| Pure ARC | ❌ No | ⚠️ Inc/dec | ⚠️ Medium | ⚠️ Needs weak |
| Borrow Checker | ✅ Yes | ❌ None | ❌ Very High | N/A (no GC) |
| Manual (current) | ❌ No | ❌ None | ✅ Low | N/A (uses GC) |

## Testing Strategy

1. **Positive tests**: Valid resource usage compiles
2. **Negative tests**: Invalid usage caught at compile time
   - Use after close
   - Double close
   - Resource leak (not closed)
   - Copy of resource
3. **Integration tests**: Real file I/O with resources
4. **Performance tests**: Verify zero runtime overhead

## Migration Path

Existing code continues to work - resource types are opt-in:

```nano
/* Old code - still works, runtime errors possible */
extern fn open(path: string) -> int
let fd: int = (open "test.txt")
(close fd)
(read fd)  // Runtime error

/* New code - compile time safety */
resource struct FileHandle { fd: int }
fn open(path: string) -> FileHandle
let f: FileHandle = (open "test.txt")
(close f)
// (read f)  // COMPILE ERROR: f already consumed
```

## References

- Clean programming language (affine types)
- Mercury language (linear/affine types)
- Rust (ownership but full borrow checker)
- Swift/Obj-C (ARC but no affine types)

