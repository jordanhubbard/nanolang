# My first exclusive borrow implementation

I continue `task_71821d84befc46e198795122c1112a27` after the shared-reference
checkpoint in PR508. I retain the full parent task and release hold: this
checkpoint supports fixed resource records with scalar numeric/bool fields,
not the complete affine or NanoISA ownership contract.

I retain an explicit `field_name` on `ASTSet` in my shared schema and both
parsers. `set view.fd value` names a field place; I do not encode a dotted
identifier or replace the whole resource. I currently allow that operation
only through an available `&mut` parameter. I check field existence and type
before native publication, and reject mutation through a shared capability.

I pass `&mut owner` as an actual mutable native pointer and preserve the
interpreter's caller-owned record identity. A mutable owner or existing
exclusive capability is required. The caller observes changes after the call;
forwarding and sequential shared reborrowing do not copy the referent. I reject
moving, storing, returning, destructuring or replacing a borrowed owner.

My ownership state uses positive call holds for shared references and a
negative hold for one exclusive reference. I reject overlap in either argument
order and reject later argument reads, moves, or field writes while an
exclusive/shared hold forbids them. I release holds when the call's argument
region ends. Distinct owners may be borrowed exclusively in one call.

My paired tests cover observable mutation, forwarding, scalar field kinds,
distinct owners, both overlap orders, moved/immutable owners, escape, wrong
fields/types and mutation in a supported later-argument match expression.
They require checker rejection rather than parser or generated-C failure and
preserve an existing output artifact. Native output assertions require actual
mutable-pointer signatures. The environment test checks exact pointer
identity, visible mutation, independent ordinary copies, and owner survival
when borrowed bindings are destroyed.

Generic and aggregate/string-bearing referents, nested field places, ordinary
record field assignment, stored references, foreign/callback borrow ABIs and
NanoISA reference lowering remain explicit unsupported boundaries. I do not
claim their completion from these native scalar-record controls. Completed
validation is recorded below after the exact-source gates finish.
