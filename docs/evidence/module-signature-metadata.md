# My compiled module signature metadata

I retain complete function parameter and result annotations when extracting
module metadata from an environment. I own those copies independently of the
parser and environment and release them with the metadata. Executable AST
pointers remain outside the compiled metadata.

My C serializer indexes every reachable TypeInfo and FunctionSignature before
emitting storage. It preserves array elements, generic arguments, callback
parameters/results, nested returned signatures and TypeInfo callback edges.
Shared pointers and cycles in the serializer input retain their identity in
the generated representation. Tuple, opaque, row and quantified-variable
fields also retain their stored values; this does not enable new language
features. Parser-produced annotation trees are copied through the existing
checked-depth ownership helpers during extraction.

I reject negative counts and missing counted storage. I escape annotation
strings as C literals and append them separately from bounded formatting
buffers. The generated representation uses static storage; it does not borrow
pointers from the compiler process. The unrelated C-text deserializer remains
an explicit unsupported API. My roundtrip compiles the generated C and checks
its initialized metadata directly.

Validation:

- `make -j8 test-module-metadata` passes all 24 C methods, both ordinary-import
  nested-array callback methods, the foreign compiler-path regression and the
  literal module-path checks.
- The strict generated-C executable checks shared/cyclic signature edges,
  nested arrays and generics, optional entries, nested returned signatures,
  tuple names, escaped strings, row metadata and quantified variables after
  the original input records are overwritten.
- The parsed callback test destroys its AST and environment before checking
  the extracted annotations, serializes them and then frees the metadata.
- A controlled extraction test performs 100 complete copy/serialize/free
  cycles. It and the graph/malformed-storage tests pass AddressSanitizer,
  UndefinedBehaviorSanitizer and LeakSanitizer without suppressions, using
  fresh `-O0` objects and the actual shared copy helper.
- The imported positive program compiles and executes; an ordinary imported
  call with a string[][] callback where int[][] is required rejects before
  replacing prior output. This depends on the complete signature comparison
  foundation from PR #486.

My parsed checker integration still exposes 595 bytes in 16 legacy checker
and environment allocations. I retain the full sanitizer log and attach it
to `task_00c47a5d65d04c48914864ec0de553d6`; no new metadata graph/copy stack
appears in that leak report. I do not claim the entire checker is leak-free.
The initial GCC 13 strict `-O1` sanitizer build also reports an unrelated
inlined `vsnprintf` null-format diagnostic in `nanocore_export.c`; that
configuration is tracked by `task_927d53891d204f2fb4e1974eb8c3edc2`. I use
fresh unsuppressed `-O0` checks for this metadata ownership boundary.

A distinct existing qualified-call check remains open under
`task_e05a42e2e09b47cc9c53fa6923eeeaef`: `cb.apply` accepts the mismatched
nested-array callback that an ordinary imported `apply` rejects. The retained
source is `/tmp/nano-imported-signatures-3tlj3nuv/bad.nano`, with its sibling
`callbacks.nano`. The original two-case qualified probe fails its rejection
assertion in `/tmp/nanolang-module-signature-integration-before.log`.
Serialization does not repair this expression-checking path.

Evidence logs: `/tmp/nanolang-module-signature-integrated.log`,
`/tmp/nanolang-metadata-sanitizer-integrated.log`,
`/tmp/nanolang-metadata-sanitizer.log`, and
`/tmp/nanolang-metadata-sanitizer-build.log`. Sanitizer objects and the bounded
harness are retained under `/tmp/nanolang-metadata-san-o0-obj` and
`/tmp/nanolang-metadata-sanitizer.c`. This is the bounded metadata prerequisite
`task_1c3c02fe49414bce919adde6666824f5`, not complete callback semantics or
native resource-callback ownership.

## My generated-list initialization repair

A fresh full compiler build after PR #487 exposed three generated `List<T>`
parameter allocations in `env_register_list_instantiation` that initialized
names and type tags but left `fn_sig` and `type_info` unset. The new metadata
copier dereferenced those indeterminate pointers. This was an initialization
bug; I did not establish a dangling AST lifetime as its cause.

I zero-initialize all three parameter arrays. I retain complete metadata
copying. My dedicated regression compiles `env.c` with a malloc wrapper that
fills allocations with `0xa5`; it fails on `parameter->fn_sig == NULL` before
the repair and passes extraction plus serialization afterward. My normal
allocator and compiler remain unchanged.

A fresh `make -j8 bootstrap` passes all native stages and the installed
compiler smoke test. The native binaries differ; this check does not claim a
fixed point. `make -j8 test-module-metadata` also passes: 24 C methods, the
poisoned-allocation check, three import methods including ordinary compilation
of `generated/compiler_contracts.nano`, and the foreign compiler-path
regression. The nested callback rejection still preserves previous output.

I retain before/after logs at `/tmp/nanolang-list-metadata-before.log` and
`/tmp/nanolang-list-metadata-after.log`, the full bootstrap at
`/tmp/nanolang-list-metadata-bootstrap.log`, and the integrated gate at
`/tmp/nanolang-list-metadata-integrated.log`. This repair is
`task_402e6b8289fc4f58b79ef5559a68dce3`.
