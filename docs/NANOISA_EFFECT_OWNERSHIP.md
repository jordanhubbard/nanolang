# My effect ownership declarations

I retain operation value-transfer signatures in ownership format version 3, extension kind 4 (`EFFECTS`), revision 1. This is a declaration prerequisite for resource-bearing effects. My affine analyzer now checks handler, perform and resume transitions against these declarations. I admit these instructions in my standalone owned value-call profile after matched VM/native cleanup qualification. My paired source producers now execute these bounded signatures; collection signatures and borrowed continuations remain incomplete.

My extension payload uses little-endian integers:

| Field | Encoding |
| --- | --- |
| Operation count | u32, 1 through 256 |
| Operation key | u32 string-pool index |
| Parameter count | u16, 0 through 256 |
| Reserved | u16 zero |
| Result | One value descriptor |
| Parameters | Parameter-count value descriptors in declaration order |

A value descriptor contains a u8 value tag, u8 zero mode, u16 zero reserved word, and u32 layout index. Scalars and strings use `NO_INDEX`. A void result also uses `NO_INDEX`; a void parameter is invalid. Records and unions require an exact retained layout of the matching kind with COMPLETE authority. Resource flags propagate through existing aggregate declaration validation. Collection and borrowed-continuation signatures require additional contracts and are not admitted by this revision.

Rows have strictly increasing key indexes. Keys must be nonempty strings without embedded zero bytes, and distinct indexes cannot repeat the same string identity. Lookup matches counted string contents, as operation dispatch matches names. Unsupported revisions, extension kinds, flags, tags, layouts, counts, trailing data and truncations fail validation.

`nvm_ownership_effect` validates the complete ownership payload before returning one signature. A missing operation or any invalid extension leaves the caller's output unchanged. An unrelated malformed global or union extension cannot be bypassed by querying an effect. Binary and explicit unverified assembly transport preserve the declarations; ordinary assembly still requires executable verification.

My [contract evidence](evidence/effect-ownership-contracts-20261011/README.md) records positive resource and nested-union signatures, capacity boundaries, mutation/truncation refusals, unchanged outputs, and the executable refusal at that earlier checkpoint. My later analysis and runtime checkpoints below establish additional boundaries; my [fresh paired source/product evidence](evidence/selected-pattern-catalog-20261011/README.md) qualifies the bounded source path under #976.

## My affine analysis

I derive each handler entry from its lexical local declarations and the retained operation signature. Parameters become fresh arm-owned values. Captured scalar initialization remains available; captured resource places cannot be read, moved, or reinitialized. Handler context participates in every state join, including the active arm and the installed-handler stack. An arm cannot pop an inherited handler.

At perform, I require exact positional argument types and transfer ownership into the operation. Every other local or staged owner and active borrow must already be resolved, since a handler can return lexically instead of resuming. At resumption, I require one value of the operation's exact declared result type and no remaining arm-owned obligations. A lexical return instead checks the enclosing function's declared result. HALT requires an empty operand stack, resolved local ownership and no active handler activation.

For bounded acyclic direct calls, I collect escaping operation identities. A callee's locally installed matching handler contains that operation; an unmatched operation propagates through callers. Calls that can escape require the caller's suspended owners to be resolved. This distinguishes locally handled effects, partially handled effects and transitive escapes without assuming all effectful calls have the same behavior.

My current analysis bounds handler depth at 64 and retains the existing bounded acyclic value-call graph. I check combined global/effect ordering with the contextual entry proof below. Borrowed captures and borrowed continuation signatures also remain outside this contract. These are remaining implementation requirements, not exemptions from the full release scope.

My [handler-flow evidence](evidence/effect-affine-flow-20261011/README.md) records positive and negative bytecode/state cases. Analysis success alone is not executable admission: the separate runtime gate checks my standalone module, exact layouts, function signatures and supported instructions.

## My owned execution

I execute admitted handlers in both the VM and my direct owned native emitter.
Native activations use bounded heap storage, alias captured scalar places, and
move argument/result owners. A handler return propagates an explicit status to
its enclosing function; each suspended native frame cleans its roots on the way.
I use no longjmp across owned frames and emit no bytecode interpreter.

My VM gives each owned handler a fresh reference generation before transferring
arguments. The first eight contexts remain preallocated; deeper effect contexts
are allocated before transfer and released on resume, lexical return, trap or VM
destruction. The execution limit is 1,024 total activations/handlers, while affine
analysis retains its 64-handler bound within each lexical context and eight
functions in an acyclic value-call graph. Runtime handler recursion can exceed
the number of distinct functions.

My [execution evidence](evidence/effect-owned-runtime-20261011/README.md) exercises
20 serialized fixtures, including resource records/unions, retained strings,
scalar/void resumptions, mutable captures, nested/recursive handlers, lexical
return through direct calls, handler-local borrows and cleanup after allocation
failure. My [fresh compiler checkpoint](evidence/selected-pattern-catalog-20261011/README.md)
qualifies C/Nano producer wiring for this bounded path. Collection signatures and borrowed continuations remain full-scope work.

## My combined global flow

My private `affine_global_flow.h` algebra separates ordinary/resumed outcomes
from lexical-return outcomes and shares initialization precondition/composition
logic with existing global analysis. My [exhaustive checks](evidence/effect-globals-20261012/README.md)
cover every encoded pair and preserve the ordinary global regression suite.
I now bind these transitions to exact persistent activation and handler stacks
from entry zero, with every global initially uninitialized. I merge initialization
states only at identical control states. Handler installation does not execute an
arm; perform enters it with the current globals. Resume restores the performing
continuation. Lexical return follows the captured normal function's caller.
I conservatively explore both branch outcomes and preserve the runtime's
1,024-activation/depth traps. Allocation failure or exhaustion of my 262,144-state
analysis budget fails closed; instruction facts remain unchanged on refusal.

My native effect context shares one global store across calls and handlers and
releases it once at the entry boundary, including lexical return and traps.
Existing global declaration and copyability restrictions remain in force.
My [integration evidence](evidence/effect-globals-20261012/integration/README.md)
qualifies ten combined source cases through both producers, 26 runtime/fault
fixtures and a fresh compiler product. Collection signatures and borrowed
continuations remain implementation requirements for the full release.
