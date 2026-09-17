# My one-parameter caller-origin contract

I implement MAC `task_48dcff3ff3314e9aad325209387968d1` after nested-reference
PR576, under affine parent ed702 and borrow parent718. I pass my
[bounded caller-origin gates](evidence/nanoisa-caller-reference.md). My multi-parameter acceptance stays
open under `task_7a2c8017c0c04b82a48ba069561e9d36`.

I admit exactly entry function 0 and one nonrecursive helper function 1. Entry
has no parameters. The helper has one shared or exclusive scalar-leaf resource
record parameter with exact retained nominal identity, scalar locals and one
int/bool/u8 result. I exclude owned helper locals, imports, indirect calls,
callbacks, recursion, deeper calls and aggregate results.

At base `925b2127`, primary opcode 0x0f is vacant in both my opcode enum and
canonical schema. I reserve CALL_REF with u32 callee and u16 reference-slot
operands. It consumes no reference value and pushes the declared scalar result.
I preserve all previous opcodes and ownership section formats 1 and 2. Existing
function parameter descriptors already carry mode and nominal layout; I do not
invent another wire schema for this call.

## My caller authority

I check a call as a reborrow of the actual caller root and immutable path,
against every active caller hold. Nominal agreement alone does not confer
authority. Shared authority cannot become exclusive; conflicting children
prevent a new call reborrow. During the call, the callee's authority suspends
conflicting parent access; return restores exactly the caller's earlier holds.

Helper reference slot 0 names the checked caller origin. Helper value local 0
is non-authoritative: ordinary LOAD/STORE and OWN operations cannot access it.
Only reference access and checked reborrows can use the parameter authority.
I preserve the actual frame identity, activation generation, owner local and
path instead of relabeling the caller place as helper local 0.

## My activation and escape boundary

I allocate two bounded reference activation contexts, not 256 descriptors for
every general VM frame. I resolve each access through the current caller frame
and stack storage. Stack resizing and core suspension retain both contexts.
A helper return expires all helper references and reactivates its caller;
terminal failure clears both contexts. No reference reaches a value stack,
heap field, returned scalar or public host-call argument. I reject public host
entry into the helper and nested host invocations while references are active.

My native private helper receives an internal descriptor for actual live caller
storage. It follows the same path and mode constraints and returns a scalar.
It neither copies a borrowed record nor writes a surrogate back later.

## My acceptance

Before widening normal verifier/runtime/native admission I require paired root
and nested field reads/mutation, compatible shared downgrade, parent restoration,
nominal/mode/conflict refusals, helper local-0 access refusal, exact joins,
non-escape at every return, terminal cleanup and yield inside the helper followed
by stack relocation/resumption. I preserve existing standalone cases and both
metadata formats, both dispatch modes, sanitizer/allocation checks and genuine
canonical compiler host linkage.

Multiple borrowed parameters require pairwise actual-place substitution,
argument-order holds and compatible shared/disjoint-exclusive alias acceptance.
That separate task cannot be closed by this one-parameter foundation. Source
production, broader calls and full affine/v5.1 acceptance remain open.
