# My self-hosted lexical capture checkpoint

I lower anonymous bodies at their lexical creation sites, resolve transitive
capture slots, retain captured value types and restore the enclosing emitter
state. I emit explicit closure environments and upvalue counts for my existing
verified VM and native closure execution paths. I preserve distinct zero-capture
instances and use tagged callable comparisons. Imported declarations retain
creation identity through their marker and merged source position after renaming.

I also preserve anonymous declarations in C-seed immediate calls, check anonymous
bodies in creation scope, and compare complete callable-expression signatures in
both direct and indirect arguments. I retain the original incorrect acceptance
of a callback returning string where int is required in
`indirect-signature-baseline.log`; the final negative matrix refuses it and keeps
prior output intact.

My compiler's recursive capture-frame save/restore exposed native record-array
writes whose fields contain tagged global array handles. I check value tag,
array storage kind and non-null handle before converting to exact field storage.
The shape constraint matches that runtime guard, including empty integer arrays.
I preserve incompatible concrete-storage refusal. Delaying the old mismatch
check alone fails; both original compiler refusals are retained.

## What I checked

- `qualified.log`: 82 callable, capture, CLI, compiler-product, lexical-scope,
  callee-snapshot, signature-metadata and native guard/lifetime test methods pass
  against the freshly built self-hosted native producer and rebuilt C seed.
- Both source producers execute the unchanged returned-closure chain, mutable
  instance aliases, independent instances, transitive callable captures, managed
  captures, function containers, imported/shadow-local captures, immediate calls,
  zero-capture identity and enclosing loop restoration in NanoVM and strict C11
  native products with ASan/UBSan/LSan. Invalid captures, scopes, arities and
  callback signatures preserve prior output.
- `tagged-fields-final.log`: three methods cover both empty constructors for all
  five supported word/string array kinds, checked set/push, at least two observed
  collections, wrong value/element/null guards, and incompatible concrete-field
  refusal with prior output retained.
- `native-regressions.log`: 2,431 structured native checks, 2,553 shape constraints
  and 379 callable constraints pass.
- `compiler-build.log`, `compiler-translate.log`, `compiler-cc.log`: the complete
  compiler source passes dependency shadows, module production, native translation
  and strict C11 compilation. The current prepared compiler is retained locally
  at `/private/tmp/nanolang-capture-producer-current` with its module and C source.
  The final C-seed indirect-signature guard was added after producing that module;
  `seed-build.log` and the final 82-method run qualify the rebuilt seed separately.

I retain early fixture mistakes (boolean assembly spelling, a test-helper call,
and record-call syntax) separately from product defects. The reduced frame source
is the exact diagnostic fixture, not an additional shadow-coverage claim.

I add the tagged-field tests to the compiler-product gate and run the capture
matrix against each newly generated compiler in both producer routes. A clean
full gate and fresh raw Stage 1/2 equality after these changes remain pending.
This checkpoint does not close my complete capture or 5.1 release acceptance:
Linux/Darwin qualification, ownership/service scope and the other release rows
remain required.
