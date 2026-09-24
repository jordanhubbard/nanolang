# My declared scalar module-artifact boundary

I restore ordinary module-manifest extern declarations under task
`task_b8838417bbc54fb98a4c49eea1b0885a`. The unchanged native module-linking suite fails
all three methods before its linker assertions: `fixture_value` is rejected by
`nisa_register_extern`. I reproduce this at repaired checkpoint `59ff072cc`.
My [transport checkpoint](evidence/pr522-implementation-2026-09-23/declared-scalar-transport) implements codec and structural checks; it does not grant executable admission.

## The current gap

My merged frontend retains the extern's owning source through `mb_owner` and
`mb_source`. `CompilerSupport.module_artifact` can build that source directory's
manifest and return the exact immutable library generation. However,
`nisa_register_artifact` recognizes only a fixed symbol/arity/result catalog,
and `nvm2c` independently requires the same kind of named adapter. Adding a
test symbol to either table would leave the language feature unsupported.

My VM already marshals declared foreign scalars through libffi for every arity.
Its signed-64, binary64, byte/bool and pointer mappings give the implementation
baseline. Calling an arbitrary native symbol through a guessed C function
pointer type is not an acceptable replacement: equal register conventions do
not establish C type compatibility.

## My wire decision

I allocate import kind 4 to `DECLARED_SCALAR_ARTIFACT`; kind 3 remains SERVICE.
This is an implementation decision, not current executable admission. I retain
the v2 import record's existing 16-byte shape: counted module/symbol indices,
signature index, kind and three zero reserved bytes. The signature table carries
every parameter and the result. I add no advisory metadata key that a consumer
could mistake for proof of source checking. Older readers must refuse the new
kind; kinds 0–3 retain their meanings and validation.

The new kind explicitly requests the scalar ABI and synchronous string-result
lifetime described below. A raw assembler may request that contract explicitly
once its complete verifier and execution support exist. The kind is not a
signature, authentication token, proof of manifest ownership, or verification
of the library's C implementation. Ordinary kind-2 artifact imports do not
silently gain arbitrary native scalar admission. Foreign declarations remain
trusted ABI claims at an unsafe boundary.

My source producer resolves the declaration's actual owning source and its
manifest into an immutable artifact generation before emitting kind 4. My
structural verifier requires a nonempty absolute counted path, a nonempty
counted symbol, no embedded NUL in either, a complete bounded scalar signature,
and no void parameters. It checks unused declarations too. The execution
loader uses that exact artifact handle and never falls back to the global
symbol search. Filesystem existence and symbol resolution are execution facts;
I do not claim that a bytecode verifier proves them from strings.

The string lifetime is part of this kind's ABI rather than a source-origin
claim: I snapshot the result before releasing arguments. I preserve the
existing provider opt-in convention, `<symbol>__nano_string_release_v1`, and
require a discovered companion to come from the called function's own image.
A missing companion means borrowed storage; a present companion receives the
original pointer once after the copy attempt. A null result remains a failure.
My VM provider cleanup now uses the typed libffi path for heterogeneous
arguments through the existing arity bound. Native kind-4 execution remains
unimplemented.
Known adapters keep their existing behavior and do not change kind implicitly.

My implementation order is transport/structural refusals, VM/native matching
execution and string cleanup, then source emission and driver/package linkage.
I require round-trip, malformed-signature, legacy-kind and linker-remapping
checks before enabling a consumer. No decoder-only checkpoint grants execution.

## Required implementation

1. I retain a declaration-derived scalar artifact contract for ordinary
   manifest-owned externs. The owner must resolve to a real manifest and an
   immutable library generation; no empty-namespace fallback or unrelated
   globally exported symbol supplies the function. Known runtime adapters keep
   their exact existing contracts. I validate every parameter and result,
   including unused declarations, before publishing a product.
2. I define one explicit ABI mapping: INT and enum carriers are signed 64-bit,
   FLOAT is binary64, BOOL/U8 use their existing byte ABI, and STRING parameters
   borrow NUL-terminated storage during a synchronous call. VOID is a result
   only. Parameter order and heterogeneous signatures survive serialization.
   I use the existing bounded FFI arity, checked exact counted symbol/path
   bytes, and complete parameter tables. Unknown tags never become integers.
   Opaque, callback, aggregate, byte-string and resource contracts retain their
   distinct requirements; this scalar step does not discharge those parents.
3. I lower admitted native scalar calls through libffi with exact storage and
   result conversion, avoiding mismatched function-pointer calls. I retain
   arguments until return conversion finishes, release them exactly once, and
   preserve evaluation order. STRING results require the existing explicit
   lifetime rule: copy borrowed results before argument release and honor an
   exact same-library release companion when present. Null and failure behavior
   must match the VM; neither a leaked owned result nor a dangling borrowed
   pointer counts as compatibility. Runtime imports and arbitrary raw module
   declarations do not silently acquire a stronger lifetime guarantee.
4. I carry required libffi headers/link flags through the actual native driver
   and installed/runtime package boundary. A generated product calls libffi
   directly and has no NanoVM dependency. Unsupported target profiles refuse
   before output replacement. My separate LLVM/Wasm host-capability work stays
   required; this C-native correction cannot close it.
5. I preserve manifest source identity, canonical-path deduplication, distinct
   sources sharing a basename, object-before-library ordering and external-root
   discovery. Building an artifact must not discard the original linker tests'
   behavior or turn them into tests of a different stub. If manifest packaging
   exposes another failure, I retain that failure and fix its owning boundary.

## Acceptance

I require all three original `tests.test_native_module_linking` methods through
fresh native stages, plus same-module VM/native checks for heterogeneous
scalar parameters/results, zero and maximum arity, source-order effects, bool
and byte tags, binary64, and string alias/lifetime behavior. Negative controls
cover wrong/missing signatures, embedded NUL names/paths, missing manifests or
symbols, unsupported ownership, and prior-output preservation. No symbol name
in the fixture has production significance.

I qualify ordinary and supported ASan/UBSan/leak configurations on Linux and
Darwin, fresh bootstrap and the existing exact artifact/lifetime suites. I
retain actual compiler, artifact and generated-product attribution. Passing
this gate is required compatibility evidence, not full FFI safety or full
LLVM/Wasm coverage. Every complete hosted partition remains required.

The first source checkpoint must specify how the serialized declaration
authority and string-result ownership are distinguished from existing raw
artifact imports. I do not widen native admission based on this prose alone.
