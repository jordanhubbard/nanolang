# My bounded mutable leaf-array adapters

I build on private boxed/packed storage and the non-admitting eligibility
analysis. I plan actual paired VM/native LLVM/import-free Wasm behavior for
ARR_NEW, ARR_PUSH, ARR_SET and ARR_POP, with generic ARR_GET/ARR_LEN. I keep
ARR_LITERAL, ARR_SLICE, nested/nominal children, cycles, imports and host/owned
contracts outside this bounded admission. They remain required follow-ons under
aggregate parent488 and managed parent51da.

## What my VM requires

I audited `src/nanovm/heap.c` and the ARR_NEW/PUSH/SET/POP/GET/LEN and STR_SPLIT
handlers in `src/nanovm/vm.c` on main `c9e2a16f`.

ARR_NEW and STR_SPLIT create capacity8 arrays. Int/float use packed width8,
bool/U8 width1; other declared kinds use boxed NanoValue width16. Creation fixes
that representation. Growth doubles capacity and refuses wrap or a new capacity
above `UINT32_MAX / element_width`. It does not saturate to UINT32_MAX. I keep
that logical capacity/growth policy distinct from the allocator's actual
metadata, native malloc calls and Wasm free-list overhead; their allocation
counts and failure thresholds are not promised identical across targets.

My private core constructors currently start without a buffer and grow from4.
I will add an explicit VM-array storage policy on private descriptors, copied
on table relocation and cleared on teardown. Adapter constructors prepare
capacity8 and preserve the declared kind before publication. Their growth uses
the VM limit and allocate-copy-commit rollback. Existing private zero-capacity
factories retain their documented behavior; private core tests do not silently
become public instruction acceptance.

Boxed VM arrays can hold different leaf tags regardless of their declared kind.
Packed writes retain their declaration and use only the already reviewed exact
matrix: matching canonical tags, U8-to-int, int-to-U8 modulo256, int-to-float
under default nearest-even rounding. Matching float payload bits remain exact.
Unsupported raw union-member fallback paths do not become runtime acceptance.

## My owner transfers

| Instruction | Adapter ownership and result |
| --- | --- |
| ARR_NEW | Publish one owned array only after descriptor/buffer preparation succeeds |
| ARR_PUSH | Consume array/value operands; retain a boxed child through borrowed append, release the operand value, transfer the same array owner as the result |
| ARR_SET | Consume array/index/value operands; validate first, retain the new boxed child before replacement, publish the edge, release the old child and operand value, transfer the same array owner |
| ARR_POP | Remove and transfer the last child owner before releasing the consumed array; return exact zero/VOID when empty |
| ARR_GET | Return a retained tagged child before normal frame cleanup releases receiver/index; return exact zero/VOID for negative or out-of-range indices |
| ARR_LEN | Read exact uint32 length as int64 before consuming the array operand |

PUSH/SET helpers consume all operands on both success and failure. Generated
FrameOutput marks those operands transferred once and tracks the result owner.
POP consumes its receiver and transfers its result. GET/LEN keep their existing
non-consuming helper/frame-consumption division. A string inserted into itself
as an existing child is retained before the old edge is released. Packed values
have no child references. Failure cannot publish a partially constructed array
or an unowned child result.

The existing private generic set is borrowed-input and returns STATE for a
missing index. That is not the VM opcode contract. The adapter first checks an
exact array receiver and exact integer index, then rejects a negative or
out-of-range full-width index without narrowing. I add `NMS_BOUNDS = 7` after
existing status values0..6, preserving their numbers. TYPE remains1 and MEMORY3.
Both status clamps (`nms_module_fail`, `nms_finish`) retain the new bounds value.
The numeric public status is my managed ABI, not a claim that the VM enum uses
the same number; tests compare the corresponding type/bounds/memory categories.

Wrong receiver/index tags still execute their checked error path even if the
analysis reports possible tag errors. All popped operands are released exactly
once. First error wins; later cleanup does not replace it. Every fallible helper
checks status before another instruction/branch executes. Callee/frame cleanup
precedes the public trapping wrapper; nontrapping status entry remains reusable.
Writes already committed to globals or shared array aliases remain visible after
later failure, as in the VM. Terminal disposal releases all remaining roots.

## My split representation boundary

Published mutable arrays must not allocate just to replace an existing element.
The private STRING_ARRAY-to-BOXED_LEAF_ARRAY promotion would introduce such an
allocation if first triggered by ARR_SET. I will prepare boxed storage during
SPLIT construction in a module that admits mutation, before the result is
published. Preferred implementation shares the byte-splitting loop but constructs
a VM-policy boxed array directly and appends tagged string children. This avoids
a second whole-buffer copy. Transactional promotion before publication is the
fallback, with its extra allocation/copy cost documented and tested.

Existing private string-only split/accessor APIs remain unchanged. Read-only
published split modules can retain their existing route; mutable-module lowering
selects the prepared boxed route consistently for every split producer. Both
routes preserve empty delimiter, stored NUL/high bytes, empty/missing values,
child ownership and exact bytes. Generic emitted GET/LEN works on either kind,
so an old string-only accessor cannot silently reject a mutated split result.
Private promotion controls remain required, but their success is not evidence
that a published allocating SET matches the VM.

## My target ABI and admission boundary

I pass scalar payload/tag inputs to packaged C helpers. GET/POP return through
explicit `uint64_t *bits` and `uint32_t *tag` outputs plus status, rather than
assuming native and wasm32 use the same aggregate-return ABI. Emitted helpers
initialize output slots, construct the LLVM tagged value only from published
outputs, and preserve exact payload bits. Target-specific packaging continues
to verify LLVM IR, hashes, no test hooks and no Wasm host imports.

For a module containing newly supported mutable-array instructions, managed
profile admission must require ordinary verification, the existing closed
profile conditions, and a successful `nvm_analyze_managed_arrays` report.
Unresolved/limit/memory results refuse publication with prior output preserved.
The analysis is a necessary storage-shape check, not authority to omit runtime
tag, bounds, lifecycle, allocation or ownership checks. Current read-only/string
modules keep their existing admission; deferred analysis opcodes do not become
an accidental regression for modules without new mutation.

I will connect the new analysis only after auditing all explicit verifier source
lists, including Makefile, host module manifests and standalone test builds.
The exact shared predicate must cover native LLVM and Wasm. No target-specific
bypass, unverified emitter route or metadata assertion substitutes for analysis.

## My implementation and acceptance order

1. Qualify VM-policy constructor/growth and prepared boxed split storage in the
   private runtime, with deterministic rollback and unchanged private factories.
2. Qualify scalar-output module adapters and consuming helper ownership, including
   bounds status propagation, exact optional results and alias/child lifetimes.
3. Add emitted helper/FrameOutput lowering and the deliberate shared admission
   conjunction, closing every new link dependency before publication.
4. Run ordinary VM/native LLVM/import-free Wasm equivalence and refusal gates.

I require all packed matrix pairs, bool/U8/integer endpoints and exact float bits;
boxed mixed leaves; alias mutation through locals, globals and calls; optional
GET/POP and missing SET; same-child replacement; empty/pop-last lifetimes;
initializer/reentry and fresh-instance distinctions; and split mutation with
stable identity and no allocating SET. Native sanitizers and deterministic
failure controls cover initial descriptor/buffer, append growth, split children,
callee error cleanup and preserved prior global/alias writes. Wasm exercises
actual finite-memory exhaustion, reuse and terminal disposal with zero imports.
I check capacity transitions8/16/32 and reject overflow using checked helper
boundaries, without attempting enormous allocations or replaying old failures.

Existing string/split/boxed/packed/core/package/profile gates remain required.
Previously blanket mutation refusals become paired admitted controls only within
this documented subset; unsupported nested/pair/opcode cases retain old-output
checks. No unchecked child shape is hidden behind a new runtime rejection.
Full arrays, nominal tracing, cycles, all applicable language transfers, Darwin
sanitizer7ba and evaluator791a remain open.
