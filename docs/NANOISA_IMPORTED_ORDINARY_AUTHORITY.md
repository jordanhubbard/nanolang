# My bounded checked-import ordinary authority (#1001)

I extend only the existing scalar/string/ordinary-record DAG declaration slice.
Arrays, generic substitutions, enum/union schemas, resources, foreign declarations,
unknown slots and executable profile expansion retain their current boundaries.
This contract precedes implementation and independent review.

## Identity before publication

My frontends already bind module-owned nominal declarations. C uses
`bind_nominal_records`; my selfhosted canonical driver uses `nb_register` and
`nb_rewrite` inside successful merged binding. A later name comparison does not
by itself demonstrate a runtime identity collision. I require their checked
identities, not guessed short names or same-shaped field tables.

My C producer retains exact AST declaration pointers at runtime registration.
For imported publication I will enumerate the root and successfully loaded
cached module ASTs, matching each declaration to its recorded runtime ordinal
by pointer identity. Repeated cached declarations count once. Missing cached
sources, an unregistered declaration, ambiguous registrations or incomplete
coverage keep optional authority absent. The original registration order and
all executable identities remain unchanged. Field references resolve only
against unique checked bound declaration names in that exact table.

My selfhosted producer will expose a separate canonical bound entrypoint, used
only after successful merged binding and type checking. The existing raw parser,
source-string and standalone emitter entrypoints retain their import omission.
The bound entrypoint validates each record's source owner and exact type binding
against the active merged-source map, including complete source-path ownership.
It carries that validation explicitly across the emitter reset for this single
invocation; no previous invocation or unrelated parser inherits admission.
Both program and selected-shadow publication use the same checked boundary.

I retain one declaration per runtime ordinal and preserve same-spelled records
from different modules as distinct checked identities. The wire layout indices
and COMPLETE flags stay unchanged. I publish no nominal slot identity that the
current producer cannot establish. Declared ordinary authority is not field-flow
proof, executable admission or permission to weaken resource validation.

## Qualification

Before positive admission I require review of the identity mapping and reset
boundary. I retain a pre-change absence baseline for existing accepted imported
source. Positive cases cover direct, transitive and repeated imports; distinct
same-spelled declarations; forward nested fields; root/dependency selected
shadows; and unchanged verified VM/native results. I compare semantic projected
layouts by source identity, allowing each producer to retain its existing ordinal
order. I require raw/unbound omission, foreign/resource/array/generic omission,
ambiguous or missing identity refusal, unchanged prior outputs on true errors,
and canonical metadata roundtrips.

I require fresh C seed and Stage1/Stage2 generations with raw fixed-point equality,
then the scoped corpus on Linux and Darwin. A C-only result or prepared compiler
from another source pin does not complete this slice. Full #1001 and row10648
remain open after this bounded imported-record checkpoint.
