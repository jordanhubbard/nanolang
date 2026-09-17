# Concrete generic union record fields

I retain concrete generic union annotations in ordinary record layouts and field
reads. This checkpoint covers MAC tasks `task_6e5fc4b3cd4f9e25eea18e792de0f2b0`
and `task_9ad126f2c5a61aadfa672f29134aa9ec` (PR #484).

## Observed failure and repair

Before this repair, a local `Outer { boxed: Box<int> }` emitted a `void*` field
through my C seed and an unspecialized `nl_Box` through my self-hosted stages.
Even an empty `Box.None {}` initializer failed native compilation. After fixing
layout, stronger reads exposed two existing frontend gaps: my self-hosted typed
read compared record-kind `Box<int>` against union-kind `Box<int>`, and my C
seed lost concrete payload metadata in a direct match on the record field.
Both read failures also reproduce with the earlier PR #475 compilers.

I now use retained field TypeInfo for concrete C spelling, instance discovery,
and dependency ordering. My self-hosted emitter consumes complete nested generic
annotations and discovers instances used only by record fields. My frontend
field reads retain declaration-aware union kind and complete arguments; direct
matches retain that metadata. Known concrete union annotations compare their
arguments as well as their declaration identity. Contextual constructors keep
their existing expected-type handling.

## Validation

At source checkpoint `3a8b17c6`, I passed:

- A fresh full `make bootstrap` with the normal shadow budget.
- 57 methods across native generic record fields, generic function values,
  global resource boundaries and generic selected ownership (220.644 seconds).
- The complete C parser and typechecker suites.
- Six adjacent map-field and nominal record-array methods, including native and
  NanoVirt execution (16.336 seconds).
- 280 parser/typechecker lifetimes under ASan and UBSan: 14 nonimport fixtures
  repeated 20 times. Existing LeakSanitizer exclusion under task `task_00c47a5d65d04c48914864ec0de553d6` remains;
  this is not a leak-freedom or runtime-GC claim.

At integrated checkpoint `6ddee9a8`, after merging main through PR #486, my
relevant C and self-hosted compiler inputs are unchanged from `3a8b17c6`.
All 25 field/callback methods passed again (126.689 seconds), including the
expanded callback foundation controls from main.

My 15 dedicated field methods execute empty/nonempty, nested `Box<Box<int>>`,
`Result<int,string>`, distinct concrete fields, field-only instances, forward
union declarations, nested ordinary records, imported records, and direct match
statement/expression cases through C seed, Stage 1 and Stage 2. Wrong concrete
arguments and wrong union declarations fail in the frontend while preserving
prior output. Resource globals and resource collections remain rejected.

Local logs use `/tmp/nanolang-generic-record-fields-` with suffixes
`bootstrap-r4.log`, `paired-r4.log`, `c-units.log`, `adjacent.log`, `asan-r4.log`
and `restack.log`. Earlier `paired.log` retains the field-read failure evidence.
The reproducible gate is `make test-native-generic-record-fields`.

## Boundary

I do not establish global resource lifetime, ownership of resource collections,
full VM parity, or generic record reflection here. Direct nested matches whose
scrutinee originates in a selected union payload remain a separate acceptance
item, `task_7bc727794375435ca72e6fef466ff161`.
