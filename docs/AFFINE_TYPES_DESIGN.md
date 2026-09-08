# Affine Ownership Contract

**Status:** Normative 5.0 design; not the current implementation

I use affine ownership with a mandatory resource-resolution obligation. An
owner may move a value, but may not copy it. Every reachable exit must resolve
each live resource by transferring it to another owner or passing it to a
consuming operation. This is not a general linear type system: a value may be
observed any number of times through a borrow, and ordinary values remain
garbage collected.

## Current Boundary

The C frontend currently recognizes `resource struct` and performs limited,
per-identifier state tracking. It has isolated positive and negative tests.
That does not establish path-sensitive ownership.

The self-hosted frontend does not yet implement this complete contract. Neither
frontend currently demonstrates all of the cases in the conformance matrix
below. Ownership metadata is not yet a verified NanoISA contract. I therefore
make no production-readiness claim for affine ownership in the current release.

Everything after this section is the 5.0 target. A target example is a
specification, not evidence that either compiler accepts it today.

## Canonical Syntax

```nano
resource struct FileHandle {
    fd: int
}

fn open_file(path: string) -> FileHandle
fn read_file(file: &FileHandle) -> string
fn close_file(file: FileHandle) -> void
```

I use one ownership spelling:

- `T` in a parameter position is by value. Passing a resource transfers it.
- `&T` is a shared borrow for the duration of one call. It cannot escape.
- `&mut T` is an exclusive borrow for the duration of one call. It cannot
  escape or overlap another borrow.
- `let next: T = current` moves a resource-bearing value. It never copies it.
- `let Aggregate { field, ... } = value` destructures an owned aggregate as one
  consuming operation. It moves every field and makes `value` unavailable.
- `drop` and `discard` are not ownership syntax in 5.0. Ordinary GC values need
  no explicit terminal operation, and resource-bearing values must be moved or
  passed to a consuming function.

The current parsers do not establish support for `&T`, `&mut T`, or owned
destructuring. Those spellings become language syntax only when both frontends
pass the same conformance cases.

## Ownership States

For each place, I track `live`, `borrowed`, or `moved`. A borrow ends at the end
of its call. A consuming call, move, or return changes the source place to
`moved`. Reading, borrowing, moving, or consuming a moved place is an error.

`resource struct` creates a cleanup obligation. A plain struct, tuple, union,
or result containing a resource is resource-bearing and inherits that
obligation recursively. These, and only these, are affine types in 5.0. All
other values are ordinary GC values and remain copyable; 5.0 has no separate
`affine` declaration. GC still manages ordinary fields inside an affine value.

## Normative Rules

### 1. Affine, Not Implicitly Linear

Ownership of a resource-bearing value cannot be duplicated or silently
abandoned. There are no non-resource affine types in 5.0 and no `drop`
operation. Ordinary GC values may be copied and need no explicit resolution.

Positive: copy an ordinary GC value; move a resource once and close the new
owner; borrow it repeatedly.

Negative: declare a non-resource `affine` type; use `drop` or `discard`; copy a
resource; use the source after a move; leave a resource unresolved at scope
exit.

### 2. Calls and Borrows

A by-value argument moves its resource-bearing argument before the callee
starts. The callee owns the parameter and must resolve it on every exit. A
shared borrow may read but not mutate, move, close, return, or store the value.
An exclusive borrow may mutate but may not move, close, return, or store it.
Borrows are call-scoped; 5.0 has no stored references or lifetime syntax.

Positive: `(read_file &file)` followed by `(close_file file)`; pass `file` by
value to a helper that closes it.

Negative: use `file` after `(close_file file)`; close through `&file`; create an
overlapping `&mut` and another borrow; return or store a borrowed reference.

### 3. Moves and Assignment

Initialization and assignment of a resource-bearing value move it. Assignment
to a place with a live obligation is rejected; the old value must first be
resolved. There is no clone operation for resources.

Positive: `let second: FileHandle = first`, then close `second`.

Negative: use `first` after that move; overwrite a live resource variable;
initialize two owners from one source.

### 4. Returns and Parameters

Returning a resource moves it to the caller. A returned resource becomes the
caller's obligation. Before any return, every other live resource owned by the
function must be resolved. Returning a borrowed value is rejected.

Positive: construct and return one handle with no other live resources; return
an owned parameter unchanged and let the caller close it.

Negative: return while a second local resource remains live; use a value after
returning it on a reachable path; return `&file` or `&mut file`.

### 5. Nested Resource Fields

An aggregate is resource-bearing when any field is resource-bearing. Moving or
consuming the aggregate transfers all nested obligations. Shared and exclusive
borrows may project nested fields. Ordinary partial moves and direct
consumption of a resource field from a live aggregate are rejected.

Owned whole-value destructuring is the terminal operation for an aggregate. It
must bind every field in one pattern; `..`, omitted fields, and refutable
patterns are rejected. The source aggregate becomes moved atomically, each
resource-bearing binding receives its field's obligation, and ordinary fields
remain GC values. The bindings must then be resolved under the normal rules.
A consuming helper can therefore deterministically dismantle an aggregate:

```nano
fn close_connection(connection: Connection) -> void {
    let Connection { socket, peer } = connection
    (close_socket socket)
}
```

`peer` needs no action because it is an ordinary GC value. Nested aggregates
may be destructured repeatedly until every resource reaches its consuming
operation.

Positive: borrow `&connection.socket` for a read; move `Connection` into
`close_connection`, destructure all fields, and close `socket`.

Negative: copy `Connection`; close `connection.socket` directly; destructure
with an omitted field or `..`; destructure and leave `socket` live; use
`connection` after destructuring.

### 6. Arrays and Collections

5.0 rejects resource-bearing element types in arrays and generic collections.
Element extraction, replacement, iteration, and destruction need a separate
place model; I do not pretend GC provides deterministic cleanup. Collections
of ordinary GC values remain unchanged.

Positive: `array<int>` and `List<string>` retain their current behavior.

Negative: `array<FileHandle>`, `List<FileHandle>`, and collections of structs
that contain `FileHandle` are type errors.

### 7. Errors and Propagation

`Result<Resource, E>` is resource-bearing only while it contains `Ok`. Pattern
matching transfers the payload into the selected arm. Each arm must resolve
the obligations it receives. `or return` and other propagation forms are
rejected when the current function owns any unrelated live resource; cleanup
must be written explicitly before returning.

Positive: match an open result, close the `Ok` handle, and return from both
arms; propagate an error before acquiring another resource.

Negative: use `or return` while a local handle is live; ignore an `Ok` resource;
move one result payload in one arm but leave it live in another.

### 8. Early Returns

Every `return` is checked independently. All resources owned at that program
point must be resolved or be the returned value.

Positive: close a handle before each early return.

Negative: close only on the final path; return early with a live nested
resource; close a value and close it again after control-flow joins.

### 9. Branches

I analyze every reachable arm. At a join, each place must have the same
ownership state in all arms. A branch may transfer ownership only if every arm
performs the same transfer or terminates after resolving its obligations.

Positive: borrow in either arm and close after the join; close in every arm and
do not use the value after the join.

Negative: close in one arm and retain in another; move to different surviving
owners across arms; omit cleanup from an implicit `else` path.

### 10. Loops

An iteration must restore every outer-owned place to its entry ownership state.
It may borrow an outer resource, but may not move or consume it in the loop.
Resources created in an iteration must be resolved before `continue`, `break`,
or the back edge. Cleanup after the loop resolves outer resources.

Positive: borrow an outer file on each iteration and close it after the loop;
create and close an iteration-local resource before the back edge.

Negative: close an outer resource conditionally in a loop; move it on one
iteration; `break` or `continue` with an iteration-local resource live.

## Conformance Matrix

Every row requires a positive and negative test in both the C and self-hosted
frontends. Passing only one frontend is not conformance.

| Rule | Positive case | Negative case |
|---|---|---|
| Declaration | resource declaration and construction | invalid resource declaration |
| By-value call | callee resolves moved argument | caller uses argument afterward |
| Shared borrow | repeated reads, then close | consume or mutate through `&T` |
| Exclusive borrow | call-scoped mutation, then close | overlap or escape `&mut T` |
| Move | move then resolve destination | use source or duplicate owner |
| Affine boundary | copy ordinary GC value | declare non-resource `affine` type |
| Drop/discard | ordinary GC scope exit needs no operation | use `drop` or `discard` |
| Return | transfer sole live resource | return with unrelated live resource |
| Nested field | whole-destructure and resolve every field | partial or incomplete destructure |
| Array | ordinary element array | resource-bearing element array |
| Generic collection | ordinary element collection | resource-bearing collection |
| Result match | resolve payload in every arm | ignore live payload in one arm |
| Error propagation | propagate before acquisition | propagate with unrelated live resource |
| Early return | resolve before every return | one leaking return path |
| Branch | identical state at join | incompatible arm states |
| Loop borrow | borrow outer value per iteration | consume outer value in loop |
| Loop local | resolve before every edge | live value at back edge, break, or continue |
| Double consume | one consuming operation | second consume |
| Scope exit | all obligations resolved | live obligation at scope exit |

For each row, the release suite must compile or reject the case with the C
frontend, repeat the same expectation with `src_nano`, and compare ownership
facts in their emitted NanoISA modules. Runtime tests may supplement these
checks; they cannot replace compile-time rejection tests.

## Release Claim

I may call this contract implemented only when all matrix rows have named tests
in both frontends and those tests pass through the product pipeline. Until then,
documentation must label examples as 5.0 target behavior and describe current
tests as partial evidence.
