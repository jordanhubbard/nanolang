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

## My initial measured checkpoint

A frozen-source bootstrap passed. The combined conformance invocation ran 63
methods: the 51 generic, module-identity, boundary, parity, import and canonical
bytecode methods passed. Its owned-pattern cases initially encountered twelve
missing-`nano_virt` setup errors in the new worktree. After building that tool,
I reran the entire affected twelve-method suite; all passed in 80.862 seconds.
I retain the failed setup log rather than call it a clean combined pass.

Resource classification and allocation-failure gates pass. All five wrapper
creation and seven publication methods pass. The 34-method compiler/backend
suite has 33 passes and one failure: compiler-bytecode-to-native translation
rejects the artifact-backed NanoISA facade import without an exact binding.
The unchanged integration baseline fails identically. MAC
`task_600074c773904b119b39bdafd85c07a5` and
`compiler-aot-artifact-binding-gap.md` preserve that required architecture gate;
I have not relaxed it or claimed full backend acceptance.

Local logs on sparky are `/tmp/nanolang-generic-affine-final-bootstrap.log`,
`/tmp/nanolang-generic-affine-final-conformance.log`,
`/tmp/nanolang-generic-affine-owned-patterns.log` and
`/tmp/nanolang-generic-affine-final-backends.log`.


## My formal-parameter scope regression

A resource record named `T` must not replace the formal parameter in
`union Box<T>`. Before this repair, all three stages reject ordinary `Box<int>`
as resource-bearing. When imported modules both declare `T`, my self-hosted
nominal binder also replaces the formal with a module-owned record identity.
I preserve the baseline in
`/tmp/nanolang-generic-formal-selfhost-baseline.log`.

I shield union formal parameters during nominal binding and unspecialized
resource classification. Concrete instantiation arguments still resolve in
the caller's scope: `Box<T>` remains resource-bearing when that caller's `T`
is a resource record. My self-hosted scoped binder handles nested type spellings;
my C union payload parser still lacks complete nested TypeInfo metadata.

Executing the positive C fixtures exposed three prerequisites. I give declared
one-letter records internal nominal names before the native emitter's legacy
free-variable heuristic. I initialize metadata slots for empty union arms so
module teardown does not release uninitialized pointers. I render an imported
generic union parameter from its concrete TypeInfo when emitting its native
prototype. The original teardown backtrace is retained at
`/tmp/nanolang-generic-formal-module/gdb.log`.

The expanded fixtures pair ordinary native copy/match execution with rejected
resource instantiations, both within a module and across same-named module
records. Rejection must retain the prior artifact. I have not admitted resource
union payload transfer, generic records, or unimplemented collection ownership.


My first formal-scope bootstrap passed, but its 21-method follow-on run exposed
four self-hosted positive-case failures: ordinary matches had no ownership
walker case whenever any resource was declared. The retained log is
`/tmp/nanolang-generic-formal-conformance.log`. I add ordinary match traversal,
arm-local scopes and joins over continuing branches. Return arms retain their
scope-exit checks. Resource-bearing scrutinees and literal payloads remain
rejected until ownership lowering exists. My self-hosted AST has no match guard
metadata; this change does not claim guard support.

New paired cases consume an outer owner in every returning arm and reject an
unresolved return arm or disagreeing continuing branches. An inline owned union
payload remains a negative case. The first expanded C-only run passes fifteen methods but exposes one further
defect: an inline owned union literal loses its union identity in the C
ownership walker and is incorrectly accepted. I retain that failing log at
`/tmp/nanolang-generic-formal-c-match.log` and repair identity lookup before
claiming the sixteen-method gate.


The next run confirmed that ordinary union constructors also lacked an
ownership-walker case. I visit their fields and retain explicit rejection of
resource-bearing payloads. I stopped the superseded conformance runner after
its confirmed failures; it is not final acceptance. The expanded C corpus now
passes all sixteen methods in 10.873 seconds.

My isolated ASan+UBSan C build passes the same sixteen methods in 10.885
seconds, with leak detection disabled. I retain two setup failures: the `-O1`
GCC instrumented build diagnoses a null format string in existing nanocore
export code under `-Werror`, and the first `-O0` invocation cannot find runtime
sources next to its `/tmp` binary. The corrected `-O0` run retains `-Werror`
and both sanitizers and supplies source/module links for that isolated binary.
Logs are `/tmp/nanolang-generic-formal-asan-build.log`,
`/tmp/nanolang-generic-formal-asan-o0.log` and
`/tmp/nanolang-generic-formal-asan-o0-corrected.log`. I do not claim a clean
initial setup or leak-checking acceptance from this result.
