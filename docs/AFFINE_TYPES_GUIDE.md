# Affine Ownership Guide

**Version:** 5.0 target
**Status:** Normative examples for an incomplete implementation

I keep resource ownership unique. Passing or assigning a resource by value
moves it. Borrowing observes it for one call. A live resource must be moved to
a new owner or passed to a consuming function on every exit.

The C frontend currently implements only basic identifier tracking. The
self-hosted frontend does not yet enforce this complete contract. The examples
below define the 5.0 target; they are not claims about current compiler support.
`AFFINE_TYPES_DESIGN.md` contains the complete rules and conformance matrix.

## The Model

```nano
resource struct FileHandle {
    fd: int
}

fn open_file(path: string) -> FileHandle
fn read_file(file: &FileHandle) -> string
fn close_file(file: FileHandle) -> void
```

- `FileHandle` marks an owned resource.
- `FileHandle` as an argument is consuming: the call takes ownership.
- `&FileHandle` is a shared, call-scoped borrow.
- `&mut FileHandle` is an exclusive, call-scoped borrow.
- Assignment and return move resource-bearing values.
- `drop value` is explicit weakening for non-resource affine values. It does
  not release a resource and is rejected for resource-bearing values.
- There is no `discard` alias.

Borrow syntax and `drop` remain target syntax until both frontends implement
and test them.

## Basic Use

```nano
fn file_size(path: string) -> int {
    let file: FileHandle = (open_file path)
    let data: string = (read_file &file)
    (close_file file)
    return (str_length data)
}

shadow file_size {
    assert (>= (file_size "data.txt") 0)
}
```

The borrow ends when `read_file` returns. `close_file` then takes ownership.
This is the positive by-value and borrow case.

The corresponding negative case is use after transfer:

```nano
fn use_after_close(path: string) -> string {
    let file: FileHandle = (open_file path)
    (close_file file)
    return (read_file &file) # rejected: file was moved
}
```

## Moving Ownership

```nano
fn move_once(path: string) -> void {
    let first: FileHandle = (open_file path)
    let second: FileHandle = first
    (close_file second)
}

shadow move_once {
    (move_once "data.txt")
}
```

`first` is unavailable after the initialization of `second`.

```nano
fn duplicate_owner(path: string) -> void {
    let first: FileHandle = (open_file path)
    let second: FileHandle = first
    (close_file first)  # rejected: first was moved
    (close_file second)
}
```

Overwriting a live owner is also rejected. Resolve the old value first.

## Explicit Drop

Affine does not mean that operating-system handles may leak quietly. `drop`
ends ownership only when no managed-resource obligation is present.

```nano
fn close_or_drop(file: FileHandle) -> void {
    (close_file file) # accepted
}

fn leak(file: FileHandle) -> void {
    drop file # rejected: FileHandle requires a consuming operation
}
```

I do not provide `discard` as another spelling. One operation is enough.

## Returning Resources

```nano
fn reopen(path: string) -> FileHandle {
    let file: FileHandle = (open_file path)
    return file
}

fn caller(path: string) -> void {
    let file: FileHandle = (reopen path)
    (close_file file)
}

shadow caller {
    (caller "data.txt")
}
```

The return moves `file`; the caller receives its obligation. A return cannot
strand another resource:

```nano
fn leaking_return(a: string, b: string) -> FileHandle {
    let first: FileHandle = (open_file a)
    let second: FileHandle = (open_file b)
    return first # rejected: second remains live
}
```

Borrowed references cannot be returned or stored.

## Nested Resources

```nano
struct Connection {
    socket: Socket,
    peer: string
}

fn inspect(connection: &Connection) -> int {
    return (send_data &connection.socket "ping")
}

fn close_connection(connection: Connection) -> void {
    # The consuming implementation resolves connection.socket.
}

fn use_connection(connection: Connection) -> void {
    let sent: int = (inspect &connection)
    (println sent)
    (close_connection connection)
}
```

`Connection` is resource-bearing because `socket` is. The whole value moves.
5.0 deliberately rejects partial moves:

```nano
fn partial(connection: Connection) -> void {
    (close_socket connection.socket) # rejected: consume the aggregate
}
```

This rule leaves no partly initialized aggregate for later code to interpret.

## Arrays and Collections

Resource-bearing elements are outside the 5.0 contract and are rejected:

```nano
let numbers: array<int> = [1, 2, 3]       # accepted
let names: List<string> = (list_new)      # accepted
let files: array<FileHandle> = []         # rejected
let files: List<FileHandle> = (list_new)  # rejected
```

The same rejection applies to collections of structs, tuples, unions, or
results that contain a resource. Element-level moves require a separate design.

## Branches and Early Returns

Every branch must produce the same ownership state at a join.

```nano
fn branch_read(file: FileHandle, verbose: bool) -> void {
    if verbose {
        let data: string = (read_file &file)
        (println data)
    } else {
        (println "quiet")
    }
    (close_file file)
}
```

Both arms retain `file`, so closing it after the join is accepted.

```nano
fn mismatched(file: FileHandle, close_now: bool) -> void {
    if close_now {
        (close_file file)
    }
    (close_file file) # rejected: file is moved on only one incoming path
}
```

Resolve before every early return:

```nano
fn early(file: FileHandle, abort: bool) -> int {
    if abort {
        (close_file file)
        return 1
    }
    (close_file file)
    return 0
}
```

Omitting the first `close_file` is the negative early-return case.

## Loops

An outer owner must have the same state at the start and end of each iteration.
Borrowing it is valid; consuming it in the loop is not.

```nano
fn read_chunks(file: FileHandle, count: int) -> int {
    let mut index: int = 0
    let mut total: int = 0
    while (< index count) {
        let chunk: string = (read_file &file)
        set total (+ total (str_length chunk))
        set index (+ index 1)
    }
    (close_file file)
    return total
}
```

```nano
fn close_in_loop(file: FileHandle, ready: bool) -> void {
    while ready {
        (close_file file) # rejected: next iteration has no owner
    }
}
```

A resource created inside a loop must be resolved before the back edge and
before every `break` or `continue`.

## Results and Error Propagation

A result containing a resource carries the obligation in its `Ok` payload.
Matching transfers that payload into its arm.

```nano
fn consume_open(result: Result<FileHandle, string>) -> int {
    match result {
        Ok { value: file } => {
            (close_file file)
            return 0
        },
        Err { error: message } => {
            (println message)
            return 1
        }
    }
}
```

`or return` is accepted before another resource is acquired. It is rejected
while an unrelated resource remains live:

```nano
fn unsafe_propagation(path: string) -> Result<int, string> {
    let file: FileHandle = (open_file path)
    let value: int = (operation) or return Err("failed")
    # rejected: the propagated error path strands file
    (close_file file)
    return Ok(value)
}
```

Write an explicit match and resolve `file` in the error arm.

## What Must Be Tested

Each rule has a positive and negative obligation:

| Rule | Positive | Negative |
|---|---|---|
| By-value call | consume once | use or consume afterward |
| Shared borrow | repeated read then close | mutate, consume, or escape borrow |
| Exclusive borrow | call-scoped mutation | overlap or escape borrow |
| Move | resolve destination | use source or overwrite live owner |
| Drop | non-resource affine drop | resource-bearing drop |
| Return | transfer sole resource | strand another resource |
| Nested resource | borrow field, consume whole | partial move or field consume |
| Collection | ordinary elements | resource-bearing elements |
| Result | resolve every payload arm | ignore payload or leaking propagation |
| Early return | resolve on each exit | one leaking exit |
| Branch | equal state at join | different states at join |
| Loop | borrow outer, resolve locals | consume outer or leak at an edge |
| Scope | all resources resolved | unresolved resource |

The release suite must run both columns through both frontends. Existing
`tests/test_affine_integration.nano`, `tests/test_resource_tracking.nano`, and
`tests/negative/resource_errors/use_after_consume.nano` are partial C-frontend
evidence only. They do not cover this table and do not establish production
readiness.

## Summary

I use affine no-copy ownership and require explicit resolution of managed
resources. By-value calls, assignment, and return move. `&T` and `&mut T`
borrow for one call. Nested resources make their aggregate affine. Resource
collections and partial moves are rejected in 5.0. Every exit, branch, loop
edge, and error path is checked. I will call this implemented when both
frontends pass the same positive and negative conformance cases, not before.
