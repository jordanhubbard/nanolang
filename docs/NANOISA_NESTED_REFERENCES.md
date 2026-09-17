# My nested reference and reborrow contract

I extend my [same-frame reference contract](NANOISA_SAME_FRAME_REFERENCES.md)
under MAC `task_556d6702b3c74377b0cb830b6d988c68`. I pass my
[bounded paired gates](evidence/nanoisa-nested-references.md). I keep one
standalone zero-argument function, an int/bool/u8 result, complete finite owned
record trees, and scalar-leaf resource referents. Caller provenance, imports,
reference parameters/results and frontend production remain separate work.

I confirm these primary slots are vacant in both my enum and canonical schema
at `765ec87b`. I preserve all earlier opcodes and operand encodings.

| Byte | Instruction | Operands |
| --- | --- | --- |
| 0x1c | BORROW_PATH_SHARED | u16 reference, u16 owner, u32 path |
| 0x1d | BORROW_PATH_EXCLUSIVE | u16 reference, u16 owner, u32 path |
| 0x1e | REBORROW_SHARED | u16 reference, u16 parent reference |
| 0x1f | REBORROW_EXCLUSIVE | u16 reference, u16 parent reference |

All four leave the value stack unchanged. I retain the existing root-only
borrow operations and reference field access. A nested borrow follows a
numeric field path through authoritative retained layouts, ending at a complete
scalar-leaf resource record. It cannot reinterpret a nominal layout. Sibling
paths can be disjoint; equal paths overlap regardless of their table indices.
I conservatively prohibit whole-owner transfer while any path is held.

## My metadata transport

My NVM container remains version 2. My OWNERSHIP section gains a separately
versioned format 2. Existing section format 1 remains byte-for-byte valid,
with no added fields. Format 2 retains the same declaration prefix, with its
version word changed to 2, then appends this little-endian table after the
last function descriptor:

- u32 path count, at most 256;
- for each path: u16 field count (1 through 32), u16 zero reserved word,
  then that many u16 field indices, followed by zero padding to four-byte
  alignment.

A format-2 empty table is valid. Root-only references use no table index.
I validate table counts, lengths, reserved bytes and trailing extent before
execution; each path instruction must reference an existing entry and resolve
against its exact owner declaration. Unused numeric entries confer no reference
authority. Assembly reconstruction retains the complete OWNERSHIP bytes;
format-1 conversion cannot erase the section. Older consumers must refuse
unknown ownership formats, never silently discard the table.

## My authority and lifetime rules

A reborrow keeps the exact root and path of its parent, requires a deeper
region, and uses a fresh reference slot. Shared authority cannot become
exclusive. An exclusive child suspends parent reads and writes; a shared child
suspends conflicting parent writes while allowing compatible reads. Ending the
child region expires its descendants and restores the parent's permitted
accesses. Exact CFG joins compare semantic paths, modes, parent slots, regions
and owner liveness. Reusing a path-table index does not manufacture an owner.

NanoVM descriptors retain owner-local and immutable path indices across core
yields. Field access resolves the current owner tree; no descriptor points into
movable value storage. Native lowering follows the same owner fields. Terminal
errors and completion clear all descriptors; suspended activations preserve
them. I do not add references to value locals, stack values or heap fields.

I rely on the verifier for overlap and parent-suspension guarantees. Runtime
descriptors implement verified traces; their fields alone do not establish
permission to execute an unverified program.

## My required gates

I require matching VM/native observations for depth-two paths, disjoint sibling
exclusive references, shared aliases, nested shared/exclusive reborrows,
parent suspension/reactivation, region reuse, branch/loop joins and owner
consumption after region end. I preserve precise refusals for overlapping
exclusive paths, shared escalation, inactive parents, same-region reborrow,
wrong field paths, incompatible joins and escaping regions. I test core resume
and stack relocation with a nested reborrow live, native mutation of the
original owner, allocation cleanup, both dispatch modes, ASan/UBSan, exact
format-1 preservation and format-2 reconstruction. I rebuild the genuine
canonical native compiler seed against the updated host module.

These gates do not establish caller alias substitution or source borrow
production. My affine/borrow parents and full v5.1 publication hold stay open.
