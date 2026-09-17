# My module-owned affine identity continuation

I reproduce this gap on the restored `e0d01acb` source boundary using the
released compiler stages: `tests/test_affine_module_identity.py` has four
methods and 24 compiler/order cases. Fourteen cases fail before repair.
The positive cases require native compilation, dependency shadows and actual
execution; source-only C emission is not acceptance. Negative cases require
an ownership diagnostic and preserve a pre-existing artifact.

My C seed rejects the second module's `Handle` as already defined. Both
self-hosted stages classify the plain module's `Handle` as resource-bearing,
then reject its legal copy/read. Reversing imports does not fix the shared
identity. These are reproducible product failures, not infrastructure incidents.
The original log is `/tmp/nanolang-affine-module-baseline.log` on sparky.

## My repair order

1. I register nominal declarations by owning module and declaration, retaining
   source spelling separately. `env_define_struct` and two registration passes
   currently query `env_get_struct`, whose fallback sees another module.
   I preserve same-module duplicate rejection, explicit imported aliases,
   nested declared-field ownership and ordinary non-colliding imports.
2. I carry canonical identity through both frontends' type annotations,
   parameter/return signatures, aggregate construction and field projection.
   My self-hosted merger currently binds functions, not nominal declarations;
   the ownership classifier's `array<string>` stores bare record/union names.
   I must not resolve a field's type in the caller's module.
3. I emit distinct aggregate names/layouts through native C, VM and AOT paths.
   My C emitter currently deduplicates composite types by `sdef->name` and uses
   spelling-based C guards. My self-hosted emitter also finds records by name.
   A checker-only change would admit ambiguous generated artifacts.
4. I run the paired module regressions, the existing ownership/classification
   gates, imported namespace gates and fresh bootstrap. I retain bytecode/AOT
   limits explicitly until those paths have passed their own fixtures.

I track this continuation in MAC `task_d7c2aa83a60b43e792fb02ae7881e397`.
The regression commit is a failing baseline, not a completed repair. It is not
wired into a release gate until implementation lands. Generic substitution,
call-scoped borrows and resource capture lowering remain separate obligations.
