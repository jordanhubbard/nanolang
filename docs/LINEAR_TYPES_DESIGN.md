# Linear Types for Resource Safety in NanoLang

**Status:** Design Phase  
**Priority:** P0  
**Bead:** nanolang-683j  
**Created:** 2025-12-29

## Problem Statement

NanoLang's garbage collector prevents memory leaks and use-after-free bugs, but it **cannot prevent semantic errors** with external resources:

```nano
fn broken_example() -> string {
    let file: FileHandle = (open "data.txt")
    (close file)
    let data: string = (read file)  // ❌ Runtime error: file already closed!
    return data
}
```

The typechecker currently allows this because `file` is just an opaque handle (int). We need compile-time tracking of resource lifetimes.

## Solution: Linear Types

**Linear types** ensure resources are used **exactly once**. Once consumed, they cannot be used again.

### Design Principles

1. **Start Simple**: Focus on external resources (files, sockets, DB connections)
2. **Explicit Annotation**: Require `resource` keyword for linear types
3. **Compiler Enforced**: Typechecker tracks usage, not runtime checks
4. **Error Messages**: Clear explanation when resource used incorrectly
5. **Dual Implementation**: Both C and NanoLang compilers must support

## Proposed Syntax

### Declaring Linear Types

```nano
/* Mark a type as requiring linear usage */
type FileHandle = resource int

type SocketHandle = resource int

type DatabaseConn = resource ptr
```

The `resource` qualifier means:
- Must be used exactly once
- Cannot be copied or aliased
- Must be consumed or explicitly released

### Functions Consuming Resources

```nano
/* Functions that consume resources mark them as 'consume' */
fn close(file: consume FileHandle) -> void {
    unsafe {
        (c_close (unwrap file))
    }
}

fn read(file: consume FileHandle) -> (string, FileHandle) {
    let data: string = ""
    unsafe {
        set data (c_read (unwrap file))
    }
    return (data, file)  /* Return resource to continue using */
}
```

### Correct Usage Pattern

```nano
fn correct_example() -> string {
    let file: FileHandle = (open "data.txt")
    let result: (string, FileHandle) = (read file)
    let data: string = (get_first result)
    let file2: FileHandle = (get_second result)
    (close file2)  /* Consumed here */
    return data
}
```

## Type System Extensions

### Type Annotations

```
Type ::= ... 
       | resource Type    /* Linear type */
       | consume Type     /* Parameter consumed by function */
```

### Typechecker Rules

**Rule 1: Single Use**
```
If x: resource T is bound, it must be used exactly once in the scope.
```

**Rule 2: No Aliasing**
```
Cannot assign resource variable to another variable:
let file: FileHandle = (open "data.txt")
let file2: FileHandle = file  /* ❌ ERROR: Cannot alias resource */
```

**Rule 3: Consumption**
```
When passing resource to function with 'consume' parameter:
- Resource is consumed (unavailable after call)
- Or: Function returns the resource (ownership transfer)
```

**Rule 4: Scope Exit**
```
All resources must be consumed before scope ends:

fn missing_close() -> void {
    let file: FileHandle = (open "data.txt")
    /* ❌ ERROR: Resource 'file' not consumed before scope exit */
}
```

### Error Messages

```
Error at line 42: Resource 'file' already consumed

  40 |     let file: FileHandle = (open "data.txt")
  41 |     (close file)
  42 |     let data: string = (read file)
                                   ^^^^
  Resource 'file' was consumed by 'close' at line 41.
  
  Hint: Resources can only be used once. If you need to read before closing:
  
    let (data, file2) = (read file)
    (close file2)
```

## Implementation Plan

### Phase 1: Typechecker Extensions (2 weeks)

**1.1 Add Resource Qualifier to Type System**
- Update `Type` enum in `nanolang.h` to include `TYPE_RESOURCE`
- Update parser to recognize `resource` keyword
- Update transpiler to handle resource types

**Files to modify:**
- `src/nanolang.h` - Type enum
- `src/lexer.c` - Add `TOKEN_RESOURCE` keyword
- `src/parser.c` - Parse `resource Type` syntax
- `src/typechecker.c` - Track resource usage
- `src/transpiler.c` - Generate correct C code

**1.2 Implement Usage Tracking**

Add to typechecker environment:
```c
typedef struct {
    char *name;
    bool is_resource;
    bool is_consumed;
    int definition_line;
    int consumption_line;
} VariableInfo;
```

Track in `check_expression()`:
- When variable is used, mark as consumed
- If already consumed, error
- At scope exit, check all resources consumed

**1.3 Add `consume` Parameter Annotation**

```nano
fn close(file: consume FileHandle) -> void
```

Parser recognizes `consume` before type in parameter list.
Typechecker marks the variable as consumed after call.

### Phase 2: Standard Library Integration (1 week)

**2.1 Update File I/O Module**

```nano
/* modules/std/fs.nano */

type FileHandle = resource int

extern fn fs_open_internal(path: string) -> int

fn open(path: string) -> FileHandle {
    let handle: int = 0
    unsafe {
        set handle (fs_open_internal path)
    }
    return (wrap_resource handle)
}

fn close(file: consume FileHandle) -> void {
    unsafe {
        (fs_close_internal (unwrap_resource file))
    }
}

fn read_line(file: consume FileHandle) -> (string, FileHandle) {
    let line: string = ""
    unsafe {
        set line (fs_read_line_internal (unwrap_resource file))
    }
    return (line, file)
}
```

**2.2 Update Network Module**

```nano
/* modules/std/net.nano */

type SocketHandle = resource int

fn connect(host: string, port: int) -> SocketHandle
fn close(socket: consume SocketHandle) -> void  
fn send(socket: consume SocketHandle, data: string) -> SocketHandle
fn receive(socket: consume SocketHandle) -> (string, SocketHandle)
```

**2.3 Add Escape Hatch for `unsafe`**

For advanced users who need to bypass linear type checking:

```nano
fn manual_control() -> void {
    let file: FileHandle = (open "data.txt")
    unsafe {
        /* Can use 'file' multiple times in unsafe block */
        let data1: string = (unsafe_read file)
        let data2: string = (unsafe_read file)
        (unsafe_close file)
    }
}
```

### Phase 3: Self-Hosted Compiler (2 weeks)

**3.1 Implement in NanoLang Typechecker**

```nano
/* src_nano/typecheck.nano */

struct VariableUsage {
    name: string,
    is_resource: bool,
    is_consumed: bool,
    def_line: int,
    use_line: int
}

fn check_resource_usage(vars: List<VariableUsage>, line: int) -> Result<void, string> {
    /* Check all resources consumed before scope exit */
    let mut i: int = 0
    while (< i (list_length vars)) {
        let var: VariableUsage = (list_get vars i)
        if (and var.is_resource (not var.is_consumed)) {
            return (Err (format "Resource '{}' not consumed before scope exit" var.name))
        }
        set i (+ i 1)
    }
    return (Ok ())
}
```

**3.2 Update Parser for `resource` and `consume`**

Mirror the C implementation in `src_nano/parser.nano`.

### Phase 4: Testing & Documentation (1 week)

**4.1 Test Suite**

```nano
/* tests/test_linear_types.nano */

/* TEST: Basic resource usage */
fn test_basic_linear() -> void {
    let file: FileHandle = (open "test.txt")
    (close file)
    assert true  /* Should compile */
}
shadow test_basic_linear {
    (test_basic_linear)
}

/* TEST: Use after consume (should fail) */
/* This should be a compile error, tested via make test */
/* 
fn test_use_after_consume() -> void {
    let file: FileHandle = (open "test.txt")
    (close file)
    (read file)  // ERROR: file already consumed
}
*/

/* TEST: Resource not consumed (should fail) */
/*
fn test_missing_close() -> void {
    let file: FileHandle = (open "test.txt")
    // ERROR: file not consumed
}
*/

/* TEST: Returning resources */
fn test_return_resource() -> void {
    let file: FileHandle = (open "test.txt")
    let (data, file2) = (read file)
    (close file2)
    assert (> (string_length data) 0)
}
shadow test_return_resource {
    (test_return_resource)
}
```

**4.2 Documentation**

Create `docs/LINEAR_TYPES.md` with:
- User guide with examples
- Common patterns (read-close, connect-send-close)
- Error message reference
- Migration guide for existing code

## Future Extensions

### Affine Types (Optional)

Allow resources to be used **at most once** (0 or 1 times):

```nano
type OptionalFile = affine FileHandle

fn maybe_open(path: string) -> OptionalFile {
    if (file_exists path) {
        return (Some (open path))
    } else {
        return None
    }
}
```

### Full Borrow Checker (Far Future)

If linear types prove insufficient, consider Rust-style borrowing:
- Track lifetimes
- Allow shared immutable borrows
- Allow single mutable borrow
- Prevent aliasing + mutation

**Cost:** 2× implementation in C + NanoLang + complex inference

## Timeline

- **Week 1-2:** Phase 1 (Typechecker extensions)
- **Week 3:** Phase 2 (stdlib integration)
- **Week 4-5:** Phase 3 (self-hosted compiler)
- **Week 6:** Phase 4 (testing & docs)

**Total:** 6 weeks for MVP

## Success Criteria

1. ✅ Compiler prevents use-after-close at compile time
2. ✅ File I/O module uses linear types
3. ✅ Network module uses linear types  
4. ✅ Self-hosted compiler supports linear types
5. ✅ Test suite has 20+ linear type tests
6. ✅ Documentation complete with examples
7. ✅ Zero performance overhead (compile-time only)

## References

- **Clean Programming Language**: Linear types for uniqueness
- **Mercury**: Modes and determinism analysis
- **Rust**: Ownership and borrowing (inspiration)
- **Linear Haskell**: Recent addition to GHC
- **ATS**: Dependent and linear types for systems programming

---

**Next Steps:**
1. Review this design doc
2. Start Phase 1 implementation
3. Update spec.json with new syntax
4. Create tracking beads for each phase

