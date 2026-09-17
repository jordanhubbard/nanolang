# My same-frame reference contract

I extend the standalone owned-transfer contract of PR564. My paired execution evidence is recorded
in [the bounded gate report](evidence/nanoisa-same-frame-references.md). MAC
`task_e31a51fc661f4102b68ad81432369a1d` depends on that merged transfer runtime.

I reserve these vacant primary bytes against source `0b8ebee8`; I do not
renumber instructions or implement the extended opcode plane.

| Byte | Instruction | Operands | Stack |
| --- | --- | --- | --- |
| 0x16 | REGION_BEGIN | none | unchanged |
| 0x17 | REGION_END | none | unchanged |
| 0x18 | BORROW_LOCAL_SHARED | u16 reference, u16 owner | unchanged |
| 0x19 | BORROW_LOCAL_EXCLUSIVE | u16 reference, u16 owner | unchanged |
| 0x1a | REF_GET | u16 reference, u16 field | push scalar |
| 0x1b | REF_SET | u16 reference, u16 field | consume scalar |

I retain OWNERSHIP and authoritative layouts. My separate reference namespace
has `local_count` slots, at most 256. A reference is never a value local, stack
token, heap field, capture, global or result. A live slot names its owner local,
region and mode. I resolve the local index on each access instead of retaining
an address into a movable VM stack. Slots survive core suspension/resumption,
but not activation completion or failure. A core `TRAP_YIELD` is resumable and
preserves every descriptor; a core execution error is terminal and clears them.
I do not claim a separate debugger resume protocol. A rejected attempt to nest
a host call does not end the suspended activation or erase its references.
I refuse nested host invocations while this standalone activation is active.

I create references only to complete scalar-leaf resource record roots with
int, bool or u8 fields. I can first unpack a nested owner into such a root.
I allow overlapping shared references, refuse exclusive overlap, refuse writes
through shared references, and require an exact field type for writes. I
forbid moving, overwriting or unpacking a held owner. Existing ordinary owner
observations cannot bypass an exclusive hold. Region end invalidates its
slots; reuse requires a new borrow. My CFG joins compare exact liveness,
region and reference identities. A return requires no live region and every
owner consumed. Branches and loops obey the same checks.

I connect these operations to `nvm_verify_affine_function` and admit them only
through `nvm_verify_owned_module` after both VM and native lowering implement
them. I preserve one zero-argument function, scalar result, no calls/imports,
no passive metadata, no reference parameters/results and no floating fields.
My native lowering accesses actual owner fields, without copy/writeback.

I require positive shared reads, exclusive mutation, region reuse, conditional
and loop access, unpacked nested owners, repeated VM invocations and core
resume, with matching native results and allocation cleanup. I retain precise
refusal tests for overlap, shared writes, wrong tags, expired slots, held-owner
transfer, incompatible joins and live regions on return; failed translation
must preserve prior output. Codec, assembly, reconstruction and computed-goto
execution must carry every opcode unchanged.

My [subsequent nested extension](NANOISA_NESTED_REFERENCES.md) implements
bounded paths and reborrows. Caller-place alias substitution, borrowed
parameters/results and source frontend production remain required follow-ups.
This root-only stage does not satisfy my full affine or v5.1 release contract.
