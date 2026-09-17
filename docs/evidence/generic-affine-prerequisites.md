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
