# My effect ownership declarations

I retain operation value-transfer signatures in ownership format version 3, extension kind 4 (`EFFECTS`), revision 1. This is a declaration prerequisite for resource-bearing effects. My affine analyzer now checks handler, perform and resume transitions against these declarations. Executable ownership admission remains closed for effect instructions until producer wiring and runtime cleanup are qualified.

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

My [contract evidence](evidence/effect-ownership-contracts-20261011/README.md) records positive resource and nested-union signatures, capacity boundaries, mutation/truncation refusals, unchanged outputs, and the existing executable refusal. Producer emission, affine handler-entry states, suspended-owner obligations, resumption transfer and runtime cleanup remain required under #976.

## My affine analysis

I derive each handler entry from its lexical local declarations and the retained operation signature. Parameters become fresh arm-owned values. Captured scalar initialization remains available; captured resource places cannot be read, moved, or reinitialized. Handler context participates in every state join, including the active arm and the installed-handler stack. An arm cannot pop an inherited handler.

At perform, I require exact positional argument types and transfer ownership into the operation. Every other local or staged owner and active borrow must already be resolved, since a handler can return lexically instead of resuming. At resumption, I require one value of the operation's exact declared result type and no remaining arm-owned obligations. A lexical return instead checks the enclosing function's declared result. HALT requires an empty operand stack, resolved local ownership and no active handler activation.

For bounded acyclic direct calls, I collect escaping operation identities. A callee's locally installed matching handler contains that operation; an unmatched operation propagates through callers. Calls that can escape require the caller's suspended owners to be resolved. This distinguishes locally handled effects, partially handled effects and transitive escapes without assuming all effectful calls have the same behavior.

My current analysis bounds handler depth at 64 and retains the existing bounded acyclic value-call graph. It refuses combined global/effect flow until handler writes and lexical unwinding have a checked global relation. Borrowed captures and borrowed continuation signatures also remain outside this contract. These are remaining implementation requirements, not exemptions from the full release scope.

My [handler-flow evidence](evidence/effect-affine-flow-20261011/README.md) records positive and negative bytecode/state cases. The runtime opcode gate still rejects effect instructions even when this analysis succeeds; analysis success alone is not executable admission.
