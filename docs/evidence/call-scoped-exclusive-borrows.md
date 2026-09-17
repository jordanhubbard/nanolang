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
claim their completion from these native scalar-record controls. My
completed checks below establish only that bounded support.

## My integrated checks

On 2026-09-17, my source checkpoint `1bc18a1e` integrates main `8e56e9cf`,
including shared borrows, the symbol lookup repair and unary-float provenance.
I regenerate my combined AST schema without losing any of those fields.

- My fresh three-stage bootstrap passes with the ordinary shadow deadline.
- My exclusive, shared, annotation and resource-callback suites pass all
  36 methods in 117.834 seconds across the C seed, Stage 1 and Stage 2.
- My parser and typechecker unit gates pass; my environment suite passes
  45 checks and the adjacent lexical-scope suite passes ten methods.
- My schema consistency check and 33 schema methods pass.
- A focused ASan/UBSan build of `env.c` and the environment test passes all
  45 checks. Other linked compiler objects are ordinary builds. I disable
  leak accounting for this focused alias/lifetime check; I do not call it
  whole-compiler sanitizer coverage.

I also guard borrowed first-class callback values before publication, including
stored, returned, forwarded and inferred/global cases. I preserve ordinary
and supported by-value owned callback controls. This closes the bounded
diagnostic discrepancy in `task_5647b9905ea3eb914389f660d54634bd` only after
reviewed integration; it does not implement a borrowed callback ABI.

Before integration, my old-base core invocation reached the ten-second shadow
deadline. I preserve `/tmp/nanolang-exclusive-core-development.log` alongside
my successful integrated `/tmp/nanolang-exclusive-bootstrap-integrated.log`;
I do not assign a new cause from those two observations. My paired, core and
focused sanitizer logs are `/tmp/nanolang-exclusive-paired-integrated.log`,
`/tmp/nanolang-exclusive-core-integrated.log` and
`/tmp/nanolang-exclusive-env-asan.log`.
