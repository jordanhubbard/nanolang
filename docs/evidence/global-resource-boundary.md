# My explicit global ownership boundary

I reject global declarations whose concrete type contains a resource. This is
an unsupported-global-lifetime diagnostic, not a global ownership model. I do
not assign the same global owner independently to every function's local flow.

Both C global registration paths call my ownership classifier after checking
the initializer. My self-hosted pass identifies nonlocal lets using the same
local/parameter/block mask as global registration, then classifies the declared
and available initializer identity. Resource-bearing empty union arms are
conservatively rejected because the declared global type can hold an owner.
An unused resource generic argument does not make an ordinary payload affine.

My paired driver tests cover mutable/immutable and inferred globals, nested
records, fixed/generic unions, empty resource union arms, collections and an
imported global. Rejections require my explicit diagnostic and preserve the
previous output artifact. Ordinary scalar/string/union globals and function
local owners remain executable controls.

## My retained generic record fields

I retain full record field annotations in my classifier. When a field contains
`Box<Handle>`, I substitute the declared union payload before propagating the
record's ownership fact. I use the same fixed point for direct nominal queries
and my `is_resource_type_info` / `has_resource_collection_type_info` APIs.
Legacy plain nominal fields can lack retained annotations; their declaration
identity remains a fallback. A recursion edge alone creates no obligation.
Ownership and collection traversal keep separate active recursion keys.

My C matrix checks twelve cases: concrete owned/ordinary/phantom arguments,
nested generics and records, collections, ordinary record and generic cycles,
a recursive resource collection, formal-name shadowing, an actual resource
record named like a formal, and a fixed union carrying a generic record.
Self-hosted shadows and paired global driver cases check corresponding resource,
ordinary, phantom, nested and collection decisions. Classification acceptance
for an ordinary generic record is not native execution evidence.

I do not extend tuple, row or function payload substitution here. Callers that
check function signatures must classify the individual parameter and result
annotations and preserve unsupported-shape guards. Resource collections remain
unsupported; these queries do not grant a collection ownership model.

## My checkpoints

At `80a61649`, I completed a fresh full bootstrap, 12 C methods, 12 Stage 1
methods, and 44 paired global/affine/selected-ownership methods. An isolated
ASan/UBSan harness repeated 11 nonimport parser/typechecker cases 20 times:
220 lifetime checks passed.

At `d170cc94`, the expanded classifier completed a fresh full bootstrap with
the normal ten-second shadow budget. Earlier classifier checkpoints reached
that deadline. A bounded measurement showed all 817 shadows passing in 9.835
seconds; I then avoided primitive nominal scans and split the large shadow
fixture into smaller declaration graphs while retaining every assertion. I did
not raise the default deadline or exempt any shadow.

At `1bf98e2c`, I integrated main through `1277bce2`, including complete C
signature storage and shared declaration ownership. A fresh full bootstrap and
the complete C typechecker suite passed. All 68 classifier/global/affine/fixed
and generic selected-ownership methods passed in 211.908 seconds. ASan/UBSan
passed 280 global boundary
lifetimes and 240 direct classifier/API lifetimes. Imported globals are covered
by driver tests. Leak detection remains disabled under
`task_00c47a5d65d04c48914864ec0de553d6`; this is not a leak-freedom claim.

At `3168885d`, I integrated the independently tested callback identity change
from PR #481. A fresh normal-budget full bootstrap, all 16 affected
classifier/global methods (25.070 seconds), and the complete C typechecker
suite passed. The C classifier and lifetime code are unchanged from the
520 sanitizer checks above. Final logs are `classify-bootstrap-final.log`,
`classify-paired-final.log` and `typechecker-final.log`.

Local logs are under `/tmp/nanolang-global-boundary-audit/`, including
`classify-bootstrap-r3.log`, `classify-bootstrap-integrated.log`,
`typechecker-integrated-complete.log`, `global-asan-integrated.log` and
`classification-asan-integrated.log`.

My ordinary local `Outer { boxed: Box<int> }` probe still fails native emission
in all three compilers, independently of globals; I track that layout under
`task_6e5fc4b3cd4f9e25eea18e792de0f2b0`. Record-global native emission remains
separate under `task_95796f5f49564ed4a911fd05a1aac5b4`. Ordinary generic union
globals provide an executable control for the boundary; I do not claim that
all ordinary global aggregate lowering is repaired.

MAC: `task_ef807591eb104cc8b664cd55581ec505` (classification) and
`task_8afaef937f934a6e9919e41b91b7a41c` (global boundary).
