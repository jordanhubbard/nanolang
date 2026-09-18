# Optional lexical local names

I retain original local names as advisory reconstruction/debugging facts. A
name does not grant access to a slot or establish its type, lifetime, ownership
or purity. With no usable name I generate a collision-free temporary name.
This bounded work implements MAC `task_d62e26f741bf47b9810a7cfa43fca44a` after
my advisory transport prerequisite; it does not complete reconstruction.

## Proposed convention

I repeat the METADATA key `nano.local.v1`. Each value is an existing string-pool
byte string containing this little-endian record, followed by a nonempty name:

| Byte offset | Meaning |
| --- | --- |
| 0 | u32 function index |
| 4 | u16 local slot |
| 6 | u16 reserved, zero |
| 8 | u32 beginning bytecode offset, relative to this function |
| 12 | u32 exclusive ending offset, relative to this function |
| 16 | exact name bytes, with length supplied by the string constant |

I use existing string limits and no new section, feature bit or opcode. The
function and slot must exist. Both interval endpoints are instruction boundaries
or the function end; beginning is no greater than end. Empty intervals retain
unused lexical declarations but never match a lookup PC. I reject overlapping
nonempty intervals for the same function/slot when interpreting this convention;
separate slots may have the same spelling over overlapping intervals. Reusing a
slot over disjoint intervals yields distinct bindings. Duplicate advisory entries
remain transportable, but ambiguous binding tables do not supply names.

My consumer validates the complete named table before offering names. An unknown
version or an unusable optional table means no original-name answer, not execution
permission or a new VM refusal. I retain its raw bytes for canonical roundtrip.
Consumers must escape or mangle exact name bytes before presenting source code.

## Producer order

I first implement a bounded codec/lookup API and assembler markers
`.local_begin <slot> "<name>"` / `.local_end <slot>` inside a function. Markers
record current byte offsets and emit no instruction. A slot cannot have two
simultaneously open marker bindings. `.end` closes remaining function-scope
bindings; unmatched ends and invalid slots are assembly errors. Generic canonical
`.metadata` output remains the lossless reconstruction form.

I then add paired ordinary C/selfhost producers for parameters and ordinary
scalar `let` bindings, including nested blocks, if/else and while scopes.
Parameters begin at function entry; a local begins after its initialization
store and ends at lexical scope exit. Temporaries without original names are
omitted. A function return may end an interval at the last emitted instruction.
The record describes lexical bytecode positions, not path-sensitive liveness.

My C producer uses its current function index and code offsets; my selfhosted
text producer uses the assembler markers rather than duplicating opcode widths.
I preserve source ownership task505's specialized producer and entry dispatch.
Pattern binders, closures/upvalues, effects, specialized borrowed profiles and
other declaration forms remain explicit later producer work unless independently
implemented and tested. No existing executable source is refused just because
its original local names are absent.

## Required acceptance

I compare parameter, shadowed-name, nested-block, conditional and loop bindings
from both ordinary source frontends against actual instruction boundaries. I
include an explicit assembly reused-slot control, since a producer may allocate
fresh slots rather than reuse them. I check exact lookup boundaries and lexical
separation, missing names, unknown versions, malformed advisory tables without
execution changes, exact name/interval and executable-code preservation through
canonical text, and unchanged VM/native results. I require byte stability on the
second canonical cycle. Existing DEBUG omission remains a separate transport
obligation; I do not claim original source artifacts roundtrip every byte.
I test allocation cleanup, normal v1 modules, metadata-bearing v1 refusal,
existing ownership/passive metadata, mandatory source shadows and the genuine
canonical compiler host build. I close only this bounded child from merged proof;
Phase20's full producer and high-level reconstruction acceptance remains open.
