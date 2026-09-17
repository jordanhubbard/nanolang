# My module-owned affine record identities

I track this repair in MAC `task_d7c2aa83a60b43e792fb02ae7881e397`.
My restored `e0d01acb` baseline failed fourteen of 24 compiler/order cases:
my C seed rejected a second module's same-named record, while my self-hosted
stages treated a plain record as resource-bearing because another module used
the same spelling. The strengthened fixture gives those records different
field names and layouts, so a shared generated type cannot pass accidentally.

## My implementation

I register C records by module owner, retain exported source names separately,
and bind colliding local record references to distinct internal identities.
Both parsers retain module-qualified record annotations. My self-hosted merger
binds record declarations and imported type aliases before classification and
native emission; its ownership facts identify declarations rather than bare
names. The native emitter uses those identities for layouts and token-derived
type annotations. Long names require complete formatted output, not a truncated
fixed-size buffer.

I preserve ordinary non-colliding record names. I reject same-module duplicate
records and collisions involving foreign records whose ABI identity I cannot
establish. I preserve an existing output artifact after those failures and
after unresolved-owner or use-after-move rejection.

## My acceptance boundary

`tests/test_affine_module_identity.py` contains nine methods and 51
compiler/order cases across my C seed, Stage 1 and Stage 2. Positive cases
compile and execute native programs and dependency/root shadows. They cover
plain copies, owned moves, nested records, both import orders, long module
names, qualified public records and two aliases for the same declaration.
Negative cases require the intended diagnostic and unchanged prior output.
Ownership rejections also require the owning source filename in diagnostics.
`make test-affine-selfhost` includes this corpus; the focused target is
`make test-affine-module-identity`.

My resource fixtures use integer payloads and terminal whole-record
destructuring. They do not establish operating-system handle cleanup. This
repair does not complete generic substitution, call-scoped borrows, resource
capture lowering, same-spelled union/enum declaration identity, recursive
nominal argument compatibility, IR ownership metadata, or bytecode fixed-point
acceptance. Existing explicit unsupported boundaries remain in force.

## My measured checks

On Linux ARM64, the integrated source includes the independent PR #379
parameter-metadata repair. A fresh bootstrap passed. All eight identity methods
(42 compiler/order cases) passed in 55.446 seconds. The existing affine parity,
contract-boundary and owned-record-pattern suites passed in a combined
41-method run. Nine existing self-hosted
module-binding methods and my C-seed module-introspection identity rejection
also passed. My focused C-seed corpus
also passed with AddressSanitizer and UndefinedBehaviorSanitizer at `-O0`,
including all eight methods, with leak detection disabled; this checks invalid
memory operations, not leak freedom. The instrumented compiler runs from its
normal repository layout so generated-module include paths resolve.

An initial custom `-O1 -Werror` sanitizer build stopped at an existing GCC
`nanocore_export.c` null-format warning. An initial relocated compiler run
could not find runtime headers. Neither attempt is counted as passing evidence;
the corrected `-O0` run retains both sanitizers and all corpus assertions.

My final combined C gates passed: resource classification, allocation-failure
handling, native transpiler and typechecker units. `make test-one-ir-compiler`
passed all 34 compiler-bytecode/AOT, map-lifetime and VM-argument methods in
249.873 seconds. These existing gates exercise the compiler implementation;
they do not establish full module-owned record equivalence for every backend.

The local logs are `/tmp/nanolang-nominal-integrated-bootstrap-r2.log`,
`/tmp/nanolang-nominal-final-42-cases.log`,
`/tmp/nanolang-nominal-integrated-regressions.log`,
`/tmp/nanolang-nominal-import-bindings.log`,
`/tmp/nanolang-nominal-introspection.log`,
`/tmp/nanolang-nominal-asan-final.log` and
`/tmp/nanolang-nominal-integrated-units.log` on sparky.

PR review found a missing `nominal_types.o` in the manual NanoVirt wrapper
link manifest. My actual foreign-module publication regression reproduced the
unresolved `bind_nominal_records` reference before the fix. After adding the
object, all five wrapper-generation and seven publication methods pass,
including foreign-module wrapper execution and daemon linking. Logs:
`/tmp/nanolang-nominal-wrapper-baseline.log` and
`/tmp/nanolang-nominal-wrapper-final.log`.

A subsequent generic annotation probe reproduced a C-seed crash in nominal
traversal: three parser allocations initialized older TypeInfo fields but left
row/type-scheme fields indeterminate. I zero-initialize those objects. The new
regression runs under allocator perturbation, executes ordinary scalar generic
annotations on all three stages, exercises the nested-array metadata allocation
on my C seed, and requires resource-generic rejection with artifact preservation
on all three stages. My self-hosted nested-generic union representation remains
a separately recorded continuation; I do not claim it from the C-only case.
The exact crash input and GDB trace are retained under
`/tmp/nanolang-generic-resource-probe/`. My revised instrumented corpus passes
all nine methods with ASan+UBSan and leak detection disabled.

The revised fresh bootstrap, all nine native methods (49 compiler/order cases
in 66.305 seconds), parser/typechecker units, five wrapper-generation and seven
publication methods pass. Final logs use `/tmp/nanolang-nominal-metadata-`
with suffixes `bootstrap.log`, `all.log`, `units.log` and `asan.log`.

The concrete-generic continuation also validates the nested-array annotation
on both self-hosted stages. The current nine-method corpus therefore has 51
compiler/order cases; the earlier 49-case checkpoint remains historical.
