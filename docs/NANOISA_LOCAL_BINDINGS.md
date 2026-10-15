# Optional lexical local names

I retain original local names as advisory reconstruction/debugging facts. A
name does not grant access to a slot or establish its type, lifetime, ownership
or purity. A future reconstruction consumer must generate a collision-free temporary name
when no usable original name exists.
This bounded work implements MAC `task_d62e26f741bf47b9810a7cfa43fca44a` after
my advisory transport prerequisite; it does not complete reconstruction.

## Convention

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
`let` bindings, including scalar, array, record, tuple, enum, union and
callback values, nested blocks, if/else and while scopes.
Parameters begin at function entry; a local begins after its initialization
store and ends at lexical scope exit. Temporaries without original names are
omitted. Both parsers explicitly mark generated destructuring owners; a user
name sharing a temporary prefix remains a source name. A function return may
end an interval at the last emitted instruction.
The record describes lexical bytecode positions, not path-sensitive liveness.

My C producer uses its current function index and code offsets; my selfhosted
text producer uses the assembler markers rather than duplicating opcode widths.
My C producer publishes function bodies in function-table order, relocating
absolute DEBUG locations before publishing passive metadata. Local-name
intervals stay function-relative. Canonical assembly retains dotted generic
specialization names without merging distinct functions.
I preserve source ownership task505's specialized producer and entry dispatch.
I retain nested-function parameters and ordinary local names in both producers,
including anonymous functions with captures. Child intervals belong to the child
function; restoring a parent keeps its bindings intact. Captured upvalue names use the separate convention below. Effects and other
declaration forms remain open unless independently implemented and tested. My specialized
borrowed profile and its scalar pattern projections are described below. No existing executable source is refused just because
its original local names are absent.

I retain legacy `par-let` binding names in both ordinary producers. A binding
begins after its initializer store and remains visible through the enclosing
scope, including after the discarded body expression. I preserve literal `_`
names and same-spelling outer bindings as separate slots. The parser retains a
distinct legacy-group flag so the typechecker checks sibling independence before
lowering the group to ordered initializers and a discarded body value. Restricted
passive and borrowed profiles do not treat this group as an ordinary lexical block.

I retain named nested-function bindings in both ordinary NanoISA producers.
Their parent binding starts after closure initialization; child parameter and
capture names remain in the child's table. The selfhost AST retains a separate
recursive self spelling, and its generated self-closure slot stays unnamed.
I preserve unambiguous C function labels and disambiguate collisions without
reserving source spellings. Direct recursive C-produced calls retain their active
capture environment in both VM execution and native translation.

## Required acceptance

I compare parameter, shadowed-name, nested-block, conditional and loop bindings
from both ordinary source frontends against actual instruction boundaries. I
include an explicit assembly reused-slot control, since a producer may allocate
fresh slots rather than reuse them. I check exact lookup boundaries and lexical
separation, missing names, unknown versions, malformed advisory tables without
execution changes, exact name/interval and executable-code preservation through
canonical text, and unchanged VM/native results. I require byte stability on the
second canonical cycle. My subsequent DEBUG transport gate also checks exact original canonical v2
bytes for the source fixtures, as recorded in `evidence/nanoisa-debug-text.md`.
This does not establish byte equality for every possible module family.
I test allocation cleanup, normal v1 modules, metadata-bearing v1 refusal,
existing ownership/passive metadata, mandatory source shadows and the genuine
canonical compiler host build. I close only this bounded child from merged proof;
Phase20's full producer and high-level reconstruction acceptance remains open.

## Specialized source borrows

I implement MAC `task_0178be8daf554a69969144b6657e7756` using the same advisory
convention. My borrowed helper parameter begins at offset zero. Its physical
value slot remains non-authoritative: only the ownership contract and reference
context describe borrowed authority. A source let begins after its initialization
store, including named owners and named destructuring projections. I omit
constructor temporaries and the parser-generated hidden destructuring owner.

I end function bindings after the emitted return and shadow bindings after that
shadow's cleanup, before the next selected shadow. Slots remain distinct even
when names repeat. Moving an owner does not shorten its lexical name interval;
these intervals never promise runtime liveness. A failed lowering publishes no
module or partial table. Stripping names leaves instructions, layouts and
ownership unchanged and must preserve verified VM/native execution.

## Captured upvalue names

I use `nano.upvalue.v1` for names in a closure environment. Its byte layout is
identical to `nano.local.v1`, but the slot is an upvalue index, checked against
that function's declared `upvalue_count`. The beginning must be zero and the
end must equal the function's code length. I reject duplicate function/index
entries when interpreting the table, including empty-function duplicates.
I keep local and capture tables independent: slot zero can describe one local
and one capture without a conflict. Unknown versions remain opaque metadata.

The assembler accepts `.upvalue_name <index> "<name>"` inside a function and
publishes a full-function interval at `.end`. Canonical text uses raw `.metadata`
records, retaining exact bytes even if the optional convention is malformed.
My C emitter records the resolved capture order after compiling a nested body;
my selfhost emitter emits names from its completed capture frame. Grandparent
captures therefore retain their source names in each intervening environment.

I expose separate `nvm_upvalue_names_validate`, `nvm_upvalue_name_at`, and
`nvm_add_upvalue_binding` APIs. They reuse the borrowed `NvmLocalBinding` record
shape but interpret the index only as an upvalue. Invalid capture tables supply
no names and never alter verifier admission or VM/native semantics. A lookup
at the exclusive function end returns absent. This metadata identifies a name
in the closure environment; it does not identify an outer declaration, prove
ownership, grant mutation rights or describe a live stack location.

## Iteration and match bindings

I begin an iteration variable's interval after its element initialization store
and end it at loop scope exit. Generated array/index/length temporaries stay
unnamed. Nested blocks, `continue`, `break`, and a later same-spelling loop do
not merge those lexical intervals. The interval describes code scope, not which
iterations execute.

I begin a union match-arm binding after storing the selected scrutinee, before
its optional guard. I end it before another arm's code. A false guard can leave
that interval without making it a liveness guarantee. Wildcard bindings remain
unnamed; an outer variable with the same spelling retains its own slot/interval.

## Effect-handler parameters

My C producer begins an effect-handler parameter name at that arm's entry and
ends it after the arm's resume. The VM installs performed arguments before
entering that interval. Sibling arms and nested handlers retain separate slots;
an outer same-spelling binding retains its own interval. Zero-argument arms
invent no parameter names. A lexical return may exit before the interval's end;
these names carry no liveness or execution authority. Self-hosted effects and
canonical native effect translation remain separate implementation gaps.
