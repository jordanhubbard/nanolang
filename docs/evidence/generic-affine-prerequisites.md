# My concrete generic ownership prerequisites

I track this continuation in `task_27d3d1bee3f84b5a9c1fc79e1f0c0748` after
my module-owned record repair. I preserve the full affine contract and the
resource-bearing collection rejection boundary.

My paired native baseline identifies these dependencies:

1. My self-hosted parser rejects explicit `Box<int>.Some` construction. An
   annotated `let boxed: Box<int> = Box.Some { ... }` uses the existing path.
2. My self-hosted checker drops concrete union arguments in
   `type_from_string_with_parser`, then returns an unsubstituted `T` from a
   match payload. An ordinary `Box<int>` read is rejected as returning `T`.
3. My self-hosted C emitter detects `array<` anywhere in a type spelling and
   emits `DynArray*` for `Box<array<int>>`. This is a representation error,
   not evidence that the source is a resource-bearing collection.
4. My C union declaration parser retains only payload kinds and names,
   discarding full nested TypeInfo. Concrete nested substitution needs that
   metadata. My C ownership pass currently rejects generic resource arguments
   conservatively; I must not remove that guard before ownership lowering.
5. My self-hosted `Box<Handle>` parameter reaches native C emission without an
   ownership diagnostic; declaration ordering then causes an unrelated missing
   record-type error. A failed native build is not the required ownership check.
6. Both frontends still need owned union payload transfer before general
   resource-generic consumption can be admitted. Classification alone does not
   establish match-arm obligations or the full `Result<Resource, E>` contract.

My first implementation slice retains concrete union arguments in checking,
substitutes payload types for ordinary executable fixtures, and rejects concrete
resource-bearing generic annotations before native publication. I retain
explicit constructor syntax, generic record declarations, unsupported nested
payload metadata and owned payload transfer as prerequisites until separately
implemented and tested. The ordinary native positive cases and diagnostic/
artifact-preserving negatives are required in both compiler stages and C.

The exploratory sources are retained under
`/tmp/nanolang-generic-resource-probe/` on sparky. The `Box<Handle>` probe also
exposed uninitialized C parser TypeInfo fields in the preceding identity patch;
that defect is repaired and tested in PR #383, not deferred here.

## My implemented classification boundary

I retain a concrete union annotation in NSType metadata while preserving its
base declaration name. Match bindings carry that annotation, and payload field
checking substitutes whole type identifiers with the corresponding arguments.
I preserve nesting when splitting arguments; similarly named identifiers are
not rewritten by substring replacement. My native type emitter recognizes an
array only when the spelling starts with `array<`, so an outer generic union
keeps its own representation.

My self-hosted ownership annotation check recursively classifies concrete union
payloads and rejects resource-bearing generic obligations before C emission.
I retain my C frontend's conservative generic-resource guard unchanged. Arrays
and other generic collections retain conservative rejection. This does not
admit resource-generic values: owned match payload transfer is still missing.
An unused generic parameter does not make a payload resource-bearing in the
classifier, but general phantom-resource acceptance is not a paired guarantee.

All eight methods in `tests/test_affine_generic_identity.py` pass across my
C seed, Stage 1 and Stage 2: 24 cases in 30.971 seconds. Ordinary integer,
string and array payloads are copied and read through native match execution;
a two-parameter union checks substitution order. Negative parameter cases
require ownership diagnostics and prior-artifact preservation for direct
resources, nested resource records and arrays of generic resource payloads.
My named tests use annotation-driven constructors already supported by both
frontends. They do not establish explicit generic constructor syntax, generic
record declarations, recursive nominal signature compatibility, complete C
payload TypeInfo preservation, or ownership metadata in bytecode.
