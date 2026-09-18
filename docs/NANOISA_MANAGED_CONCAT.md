# My managed concatenation and frame contract

I record this implementation contract under
`task_b1cc086f8cdf476cb0814f5ade9a15b1`, a child of managed-runtime task
`task_51da49b39230468784da3481b893563b`. My runtime core is merged; this
contract does not admit additional instructions by itself.

## Roots and consuming operations

Each occupied operand-stack slot, local and global owns one reference when its
value has my string tag and a dynamic handle. Literal handles need no retain.
LOAD and DUP retain before publishing the additional value. STORE transfers a
popped value and releases the previous destination. POP releases. SWAP moves.
CALL transfers argument owners into callee locals; return moves its result out
before releasing every remaining local and stack owner. Uninitialized slots
are void. Branches preserve exactly the owners represented by live slots.

My consuming concat helper takes one reference per input, even when both
handles are equal. It copies both byte sequences into one new allocation
before descriptor growth can invalidate descriptor addresses. It checks the
combined length before allocation, preserves embedded NUL bytes, and writes
one trailing terminator outside the logical length. It publishes one result
owner only after allocation succeeds. It releases both transferred inputs on
success or error; other retained aliases remain valid. Failure leaves the
output argument unchanged, but deliberately consumes the input owners.

STR_CONCAT requires two strings. Generic ADD retains numeric behavior and
concatenates two strings; mixed string/numeric operands report TYPE. Every
consumed operand is released, including operands rejected by tag checks.
Substring and conversions remain refused until their separate prerequisites
are implemented; substring task `task_ce840367841a4bdb94ab69fd2446b635` stays open.

## Error and instance boundaries

Every managed emitted function returns an internal value plus status. Numeric
checks, assertions, helper failures and return checks branch to cleanup rather
than trapping inside a live frame. A failing callee has consumed its arguments;
the caller releases its remaining roots and propagates status. Successful
returns transfer their single result owner to the caller. No result owner is
published on failure. Earlier global writes persist; a failed instruction does
not replace its destination. Globals survive repeated entries on one instance.

`nano_try_entry` guards reentry and returns packed status/result after frame
cleanup. The existing entry wrapper traps only after cleanup. `nano_dispose`
releases global roots and then disposes runtime storage; disposal while active
returns BUSY. Disposal is terminal and idempotent. Native executable main
disposes on both success and reported failure. Reusable native and Wasm hosts
explicitly dispose their instance. Engine faults, process termination and
machine-stack exhaustion remain outside recoverable language errors.

## Runtime packaging proposal

I generate target-specific LLVM runtime IR from the canonical portable C core
at translator build time, using an explicit Clang tool dependency. I embed
native-64 and wasm32 variants in the translator. Emitted `.ll` contains the
selected runtime definitions and application definitions; it needs no hidden
project object when executed or linked. I validate the combined IR, target
layout and symbol namespace before enabling this path. Existing API callers
default to native; Wasm packaging explicitly selects wasm32.

I do not cast the existing `{ptr,i64}` literal descriptor to the C core's
`{ptr,uint32}` view. A checked generated descriptor adapter and scalar-argument
ABI avoid depending on C aggregate calling conventions. Runtime context stays
private to one emitted module instance. Native support requires the declared
64-bit pointer/size ABI and malloc/free; Wasm uses my reclaiming allocator and
has no libc, WASI or other undeclared imports. Runtime helper prefixes `nms_`
and `nano_runtime_`, plus `nano_try_entry` and `nano_dispose`, are reserved for
custom entry validation. I retain original scalar/literal admission until
these packaging and cleanup checks pass together.

## Ordered acceptance

1. I test consuming concat without changing executable admission: literal and
   managed aliases, equal handles, empty/NUL bytes, table growth, deterministic
   byte/table allocation failures, native sanitizers and Wasm storage reuse.
2. I review and test target IR packaging independently, including production
   builds without test hooks, combined IR verification, reserved entry refusal
   with output preservation, and zero Wasm imports.
3. I wire stack/local/global/call ownership and status cleanup before enabling
   STR_CONCAT or string ADD. I compare normal VM, LLVM and Wasm results, nested
   calls, branch joins, loops, repeated/fresh instances, assertions and type
   errors, OOM rollback, live allocation counts and final disposal.
4. I update profile documentation only after the paired gate passes. The full
   managed-string and applicable-language parents remain open.
