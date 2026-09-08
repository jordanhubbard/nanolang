# Affine Types for Resource Safety

**Status**: 5.0 design contract; the current C-seed MVP is partial
**Roadmap**: Phase 20 in `ROADMAP.md`
**Tasks**: contract `task_4ac22044ffda9f93b336a85573293bc2`, C seed
`task_c4e2f078cef8c4e461f0de3711c8a2b9`, self-hosted frontend
`task_20048de825616195b9f2bc492231a851`, NanoISA
`task_ed70242ac4d83be7b2327da7ece387ad`, services
`task_d03c232dc067e75cbc2fb2b7fb84ee46`, release gate
`task_28f2fb4b1f3c8a5ce93df628bb569d76`
**Direction**: affine ownership for resources + GC for ordinary values

This document describes my 5.0 target, not the behavior I can prove today.
The C seed recognizes `resource struct` and has a basic per-identifier
unused/used/consumed tracker. That prototype is not path-sensitive and does
not prove moves, nested ownership, all exits, loops, or equivalent behavior in
`src_nano`. Ownership metadata is not yet a verified `.nvm` contract. I will
not call affine ownership production-ready until the task-backed acceptance
matrix above passes.

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
A type is **affine** if ownership cannot be duplicated and an owned value can
be transferred at most once. Non-consuming observation, if I accept borrowing,
does not transfer ownership. After a move, consuming call, or explicit discard,
the previous owner cannot access the value.

Managed resources add a cleanup obligation: every reachable exit must either
transfer ownership or perform an ownership-ending operation. This obligation
is closer to a linear lifecycle than plain affine weakening. The 5.0 semantic
contract task will settle the terminology and exact `drop` behavior before I
encode it in either compiler.

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

`&` is proposed 5.0 syntax, not current implementation evidence. The contract
task must either accept it and define its lifetime/aliasing limits or replace
it consistently in this document and the guide. Passing a resource by value
always transfers ownership.

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

## 5.0 Implementation Plan

I execute this in dependency order; the MAC task ledger is canonical.

- [ ] Freeze one semantic contract and canonicalize both affine documents
      (`task_4ac22044ffda9f93b336a85573293bc2`).
- [ ] Implement path-sensitive checking in the C seed
      (`task_c4e2f078cef8c4e461f0de3711c8a2b9`).
- [ ] Implement the same syntax and ownership decisions in `src_nano`
      (`task_20048de825616195b9f2bc492231a851`).
- [ ] Emit, serialize, link, reconstruct, and verify affine facts in NanoISA v2
      (`task_ed70242ac4d83be7b2327da7ece387ad`).
- [ ] Migrate real standard-library and NSI service handles
      (`task_d03c232dc067e75cbc2fb2b7fb84ee46`).
- [ ] Pass the dual-frontend, NanoVM, and AOT C acceptance matrix
      (`task_28f2fb4b1f3c8a5ce93df628bb569d76`).

The older week-based checklist mixed a parser prototype with a completed
language guarantee and assumed the AST-to-C transpiler remained the product
backend. Phase 20 instead requires both frontends to emit one verified `.nvm`
product; translators cannot repair ownership facts discarded by a frontend.

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

## Contract Decisions and Later Extensions

### Borrowing (5.0 contract decision)
Temporary access without consuming is required by the guide's repeated-read
examples. It is therefore a contract decision, not an optional enhancement:

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

### Explicit Drop (5.0 contract decision)
If I permit affine weakening, explicit discard must still define what happens
to the underlying resource. The contract must distinguish a cleanup operation
from abandoning a live handle:

```nano
fn use_file() {
    let f: FileHandle = (open "test.txt")
    let data: string = (read_all f)
    drop f  // Explicitly consume without calling close
}
```

### Resource Pools (after container ownership is defined)
Pools are outside the 5.0 affine release gate unless the semantic contract
accepts resource-bearing collections and the verifier can represent their
element ownership. This example is illustrative, not supported syntax:

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
