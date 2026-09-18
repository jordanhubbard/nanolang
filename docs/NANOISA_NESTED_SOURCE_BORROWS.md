# My bounded nested source-borrow contract

I track this slice as MAC `task_91cb8c2db94941bc950d9e9fa597d77a`, after my
[multiple-formal source producer](NANOISA_MULTI_SOURCE_BORROWS.md). My affine
and borrow parents remain open. I consume my existing checked nested-path and
multi-caller runtime contracts; I add no runtime authority or wire format.

I admit finite plain resource-record trees in declaration dependency order:
each child record must precede its parent. This excludes cycles and forward
layout dependencies. My leaves contain only `int` and `bool` fields. A path
crosses at most 32 nested record fields; a scalar-leaf record has depth zero.
I retain the existing bounds of 256 local slots, 256 record fields and 256
numeric path records.

A nested constructor moves an existing, exactly declared child owner once.
I evaluate fields in source order into unnamed typed temporaries, then pack
in declaration order. I do not yet admit inline nested literals. A nested
record pattern names every field exactly once, in any source order. I unpack
its parent shell once into unnamed field slots, then transfer named child
owners or copy scalar projections. Moves retain exact nominal identity.

My source checker still requires every named resource to be consumed. I do
not introduce implicit source drops or explicit discard syntax. My lowering
cleanup can drain a remaining complete tree using `OWN_UNPACK_LOCAL` and
`OWN_STORE_LOCAL`; this is defensive machinery, not admission of a checked
program that leaves a live named tree at scope exit. Accepted nested source
uses complete construction and destructuring. I invent neither destructors
nor record copies.

My program still has entry zero and helper one. The helper accepts one through
eight borrowed scalar-leaf record formals, with scalar remaining locals and
result. I resolve a root-local field chain to declaration-index numeric paths.
Modules with paths use ownership format two; root-only modules retain format
one. Argument preparation remains left to right in one region, retaining
earlier holds until the call completes. I compare actual roots and path
prefixes: shared aliases and disjoint siblings are allowed; any overlapping
pair containing exclusive authority is refused. My verifier checks actual
caller-origin substitution before execution.

Direct scalar reads require a leaf-record referent. A mixed container's scalar
fields can be destructured, but direct nonleaf borrowing remains outside this
slice. Helper mutation remains leaf-only. I do not store or return references.
Advisory local names retain their existing lexical intervals; compiler-created
constructor and unpack slots remain unnamed.

I require exact paired C/selfhost layouts, ownership paths, code, names and
canonical text. Every selected shadow must lower, or I refuse publication
without overwriting an accepted output. Positive VM and sanitized native
checks cover reordered construction and patterns, repeated calls, distinct
nominal records, shared aliases, disjoint exclusive siblings and observable
mutation. Ordinary refusal checks retain nominal, move, overlap, path and
source-lifetime boundaries. Control flow, imports, deeper call graphs, inline
aggregate construction and the broader ownership roadmap remain separate.
