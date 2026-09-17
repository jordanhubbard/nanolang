# I scope native record temporary storage to each call

My emitted C kept every record temporary on the native stack for the entire
function. On this macOS arm64 host, Clang's `-O0 -fstack-usage` reported these
frame sizes before and after moving that pool:

| Function | Before (bytes) | After (bytes) |
| --- | ---: | ---: |
| `parser_with_position` | 333,296 | 5,712 |
| `parse_primary` | 3,364,768 | 1,627,872 |
| `generate_expression` | 5,381,648 | 2,077,664 |

I allocate the record temporary pool once per invocation using checked
`calloc`. Every normal return and halt snapshots its result, then reaches one
cleanup path that frees the pool. Cross-function tail calls finish their call
before that cleanup; they still use the C stack. Self-tail restarts reuse their
current allocation. Returned records remain value copies, and separately owned
nested records and array handles do not point into the freed pool.

I have not eliminated large local records or implicit C argument copies.
The remaining megabyte-scale frames prevent a general bounded-recursion claim.
MAC `task_ec6559ac267b475daec33dd711c34ec1` tracks that work. Pool sizing also
still follows temporary high-water counts, not a liveness allocator.

## I test lifetime and stack behavior

My new test emits 300 record temporaries with a 75-field record width and runs
twelve recursive steps with a 2 MiB stack cap. Ordinary calls, self-tail calls
and cross-tail calls each run ten times. Allocation instrumentation checks zero
live frame allocations after each return, exact single-allocation reuse for
self-tail calls, and unchanged returned record contents. Three injected
allocation failures must trap. These cases pass under strict C11 at `-O0`.

My existing high-water tests now require `calloc(300, sizeof *r)` for records
instead of a 300-element stack array; their execution assertions remain.

`make test-nvm2c` passes 1,670 AOT and 1,073 shape checks.
`make test-nvm2c-sanitizers` passes the same counts with fresh ASan/UBSan
translator and shape objects. That gate disables leak detection; the focused
frame test separately counts frame allocations and frees.
`make test-one-ir-compiler` passes 18 of 19 methods; only full compiler execution
remains failed. `git diff --check` passes. I do not treat those focused checks
as proof of all native record lifetimes or general recursion bounds.

## I locate the next execution failure

A native compiler run now gets through parsing and aborts in `nsarr_push`,
called by `cg_append` while `gen_c_runtime` emits C. The fixed string-array arena
cannot hold the accumulated append allocations. MAC
`task_7b8691dd087e48ea9afdeb89ddcf4640` tracks owned capacity growth, not a larger
fixed arena.
At the failing append, the array has 330 elements and the arena has consumed
65,504 of its 65,536 pointer slots. Each append copies the prior contents into
new arena space, so this is accumulated allocation, not a 65,536-element array.

I also explained the previous differing signals: direct commands have an
8,176 KiB stack limit; this host's `make` gives recipes 65,520 KiB. I checked the
latter with an in-memory makefile whose recipe runs `ulimit -s`. Direct runs hit
the old stack overflow, while the larger-stack recipe reached the arena abort.
I do not raise either limit in the implementation.

The parent task `task_e81212c768d148639429d1d2be9f826d` remains incomplete because
record locals and argument frames still require work. This is partial release
progress, not compiler acceptance or a recursion-space proof.
