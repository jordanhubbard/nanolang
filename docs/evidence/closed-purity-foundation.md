# My closed purity foundation

I retain `pure fn` in my selfhost AST schema, including both canonical function
binding and nominal type rewriting. Both frontends now traverse every reachable
known function body, with a per-root visited set. Recursive cycles cannot hide
an effect reached elsewhere in the cycle. I derive summaries from bodies even
when a helper has no annotation; an extern annotation alone supplies no proof.

I share explicit intrinsic and observable-host classifications through
`spec/purity_intrinsics.json` and checked generated C/NanoLang tables. Callback
intrinsics and GPU state queries are not in the closed intrinsic set. Unknown
calls, computed calls and unclassified constructs remain open. Mutable reads,
mutation, unsafe blocks and resource-bearing signatures reject a pure claim.
Loops and local mutation retain the existing conservative rejection policy.
An immutable global array/map/list handle is not deep-immutability evidence.
I reject reads of such globals and refuse mutable aggregate parameter shapes,
including plain records that contain those handles. Only scalar/string/enum
values and recursively scalar/string-only records are admitted as external
inputs in this slice; tuples, unions, opaque and unclassified inputs remain
conservative. Private array construction is still allowed.
Ordinary private GC allocation remains accepted for supported immutable values.

I do not export this analysis as trusted NanoISA metadata. The `par`/`flow`
syntax, place/dependency analysis and complete eligibility proof remain open.
This change is a frontend prerequisite, not the passive-parallelism cutover.

My shared conformance test compiles six positive and twenty negative programs
with Cseed and Stage 2. It also checks a bound imported helper before and after
adding an observable effect. Positive native programs execute their assertions;
negative cases require the closed-summary diagnostic and preserve prior output.
The fixtures cover recursive closure, unannotated helpers, private strings and
records, arrays and lexical shadowing, immutable and mutable globals, hidden
record-field effects, callback shadowing, unsafe/extern boundaries, higher-order
calls, loops and resource transfer/return/wrapping.

Validation commands:

```sh
make -j4 bootstrap test-typechecker test-runtime-lists
MALLOC_PERTURB_=165 python3 tests/test_purity_contract.py
make schema-check
python3 scripts/gen_purity_intrinsics.py --check
```

I also compiled and ran `test_complex`, `test_quaternion`,
`test_std_math_extended` and `test_std_math_extended_full` with Cseed. My generated
ASTFunction layout is exercised by the bootstrap and all runtime list tests.

Resource-signature inspection exposed an existing imported-parameter copy bug:
module registration omitted `Parameter.type_info`. I now preserve all borrowed
parameter metadata before duplicating owned names. The malloc-perturbed imported
fixture checks this path.

Stage 2 shadows also exposed the separate native bool-array record-field setter
bug, tracked as `task_d32adbdff13241dc8ad9b0a889071352`. My analysis uses explicitly
typed array locals for those mutations. I retained every mandatory shadow and
its assertions; the underlying emitter followup remains open.

I track this foundation as `task_41966fd9c9da4f1babfab0a8a25a66c4`.

I construct qualified call names with standard C99 `malloc`/`snprintf`; I do not
require a platform-specific `asprintf` declaration in this analysis.

I reject explicit extern declarations that reuse intrinsic names, including
`abs` in an unsafe module. Both frontends inspect the declaration before
using the intrinsic contract. My immutable-map negative uses a typed local
factory, so it reaches this purity check without a constructor-context error.

My `complex_exp` wrapper retains its behavior and shadows but no longer claims
a closed summary for its explicit foreign `exp` dependency. Verified foreign
intrinsic binding contracts remain a full-scope continuation, tracked as
`task_20f6cb36fbf24bba987b4ea503529438`.
