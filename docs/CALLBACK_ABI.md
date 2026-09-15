# My retained callback ABI

## Status

I require this work before 5.0. This document defines my implementation
contract; it is not evidence that my compiler or dispatch module implements
it yet. I keep those acceptance items on my roadmap.

My handle runtime in `src/runtime/callback_runtime.c` passes
`make test-callback-runtime`: 40,000 cross-thread invocations, scalar signature
checks, same-owner nesting, failed allocation/publication, owner-only cleanup,
deterministic queued cancellation, and late calls after shutdown. Both test
binaries pass AddressSanitizer/UndefinedBehaviorSanitizer and ThreadSanitizer
on Darwin arm64. LeakSanitizer is unsupported on that host; I ran ASan with
`detect_leaks=0`. The existing 22 FFI unit tests still pass. None of this
establishes callback-import metadata, VM activation, dispatch, or COP integration.

I now preserve declared ordinary function parameter tags in the execution
module and through v2 serialization. Legacy producers still leave unknown
tags; an unknown tag is not permission to publish a callback. My v2 bridge
passes 295 checks, including mixed parameter order, distinct-signature
interning, and table growth. Allocation-failure tests check transactional
growth and updates, including a source entry borrowed from the growing table.
They pass ASan/UBSan with leak detection disabled on Darwin.

The 64-check NanoVirt suite includes reused parameter names across nested
functions, distinct nested result types, a void capturing closure, and a
tail-position call that must preserve captures. Typechecker tests reject a
nested result of the wrong type and a nested `break` targeting the outer
function's loop. The 93-check verifier suite passes. This is declaration
preservation and selected execution evidence, not full callback support.

## Boundary

I do not turn a bytecode function index into a machine address. A callback-
aware native adapter receives a versioned handle with an immutable signature
and `retain`, `release`, and `invoke` operations. An ordinary C function-pointer
parameter is a different ABI. I require an explicit adapter contract at the
import boundary, checked against the function's declared signature. I do not
infer that contract from names such as `nl_queue_async`.

My initial handle ABI carries signed 64-bit integers, doubles, booleans,
bytes, borrowed opaque pointers, and void results. I reject unsupported tags
and mismatched argument or result types. Strings and aggregates need explicit
marshalling and ownership contracts before crossing this boundary.

## Ownership and shutdown

I lend a callback handle for the duration of a foreign call. A native adapter
retains it before publishing asynchronous work and releases it after its last
possible invocation, including cancelled work. Every invocation borrows a
live reference for its entire duration. Retaining an already released pointer
is invalid C, not something a reference counter can repair.

I root the callable and captures while the handle is live. Only the VM's
owning thread creates or destroys those roots. Releasing the last native
reference queues reclamation for that thread. Failed publication returns
ownership to the submitter instead of leaking the root.

Shutdown first closes admission and cancels queued invocations. On the owner
thread I detach every VM root before destroying the heap. Retained native
handles remain valid tombstones: `invoke` returns cancellation without reading
the former VM. They free themselves after the final native release. I neither
wait indefinitely for a distant timer nor free a trampoline still held by C.
Shutdown during an executing callback is rejected; the host retries after
unwinding. Cancellation is an explicit result, not successful execution.

## Execution

Native threads enqueue typed requests and wait for their result. They never
read or mutate VM frames, globals, reference counts, or intern tables. The
owner services requests at safe points. A same-owner invocation may nest;
each nested activation must restore the suspended activation on every exit.

Foreign calls that can wait for callbacks must run outside the owner while
the owner pumps requests. Marshalling and result construction remain owner-
thread work. Moving the existing whole FFI function to a worker would race
heap allocation and is not an implementation of this contract. Native code
that requires a particular OS thread needs an explicit execution policy.

I serialize VM execution, not native work. Native concurrent queues still
schedule asynchronously; callbacks share the original globals and heap. I do
not simulate completion by running every submission immediately.

## Co-process boundary

Pointers and handle vtables cannot cross a process boundary. Isolated calls
need registered callback IDs, typed request/result messages, cancellation,
and retain/release accounting. Until that transport is implemented, a
callback-bearing isolated import must fail explicitly before publication;
I must not silently fall back to in-process execution. The final release
acceptance must state and test the supported policy.

## Evidence required

I test native retention beyond the submitting call; concurrent invocation;
owner-only execution and destruction; nested invocation; exact signatures;
callback errors; final release during execution; cancelled queued calls;
late invocation after shutdown; and allocation/publication failures. VM
integration adds closure capture collection, shared globals, nested foreign
waits, instruction safe points, and trap unwinding. Dispatch acceptance keeps
all dependency shadows and delayed queue/group work enabled.
