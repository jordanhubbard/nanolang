# I attach generated typed provider images explicitly

I keep the private description checkpoint separate from executable admission. This next contract makes the concrete loader and adapter interfaces reviewable before I implement them. It does not enable a wire feature, replace existing scalar adapters, or permit a private native compiler fallback.

## I anchor an image to its retained generation

My existing module builder publishes an immutable `.nano-gen-*` directory and switches `current` only after complete staging and barriers. I resolve and retain the exact generation before attachment; I never resolve `current` again to establish call identity. Installed SDK validation already checks required file hashes and roles, but that SDK inventory is not itself a foreign provider manifest. A generated provider manifest must separately identify its exact library, complete declaration/lifetime bytes, target ABI, required dependency artifacts and generated adapter schema.

I will add an owning generation/image attachment object, keyed by the verified generation identity and physical artifact path rather than a logical short module name. Description preparation remains nonloading. Attachment requires the exact immutable manifest entry, verifies its library and schema digests, and stages all loader references and descriptor storage before publication. A failed attachment leaves the caller's old generation and output unchanged and releases only its own staged references. Native image constructors may execute during loading; this is trusted native loading, not a rollback promise for arbitrary constructor side effects.

`dlsym(handle, name)` may return a dependency symbol. My existing string-release and array-ABI helpers compare function/companion `dladdr` bases, which is necessary but does not anchor either to the intended artifact. New attachment must identify the loaded schema anchor's actual image path, resolve it to the verified retained generation artifact, and compare the generated adapter and every pin/drop/callback hook against that image base. A matching pair of symbols from a dependency refuses. Missing image attribution also refuses. I keep the image reference alive through all descriptors, callback handles and provider-owned results. I do not add global symbol fallback or infer owner from a unique symbol spelling.

The immutability condition is an explicit module-generation lifetime contract. The existing builder does not mutate published generations. For externally supplied mutable paths, attachment must first obtain an independently owned immutable generation with its declared dependency closure or refuse; a pathname hash followed by a later unrelated `dlopen` is insufficient. I must qualify source-hidden and read-only installed paths without writing into the installation.

## I call C through generated exact prototypes

The schema anchor is a versioned generated data object. It contains bounded row counts, exact declaration/lifetime digest, target ABI identifier and adapter/hook symbol names. It is not an arbitrary function-pointer table accepted on spelling alone. Its scalar header and row lengths are checked before traversal. Generated provider C includes the actual SDK headers and emits static layout assertions for the selected record/union/tuple ABI. The compiler checks each actual exported prototype in that translation unit.

Each import receives a generated value-envelope adapter with one shared, explicitly defined transport prototype. That adapter validates the exact signature/schema identifier and envelope bounds, decodes each value into its actual declared C type, then calls the provider using its real prototype. It writes the result through typed fields into the bounded neutral envelope. VM and AOT select the same generated schema and adapter; neither casts an arbitrary aggregate function through an integer-return ABI. Existing scalar-only descriptors remain unchanged.

The neutral envelope carries existing shared type indices and bounded node identifiers, not C addresses or a second type system. Record/tuple fields preserve declaration order; unions preserve exact variant ordinal and that variant's declared fields; arrays carry count plus child references; scalar floats preserve their defined bits. Every edge is checked against the retained exact type graph and policy path before foreign entry. A prepared call reserves argument decoding, result capture metadata, logical-owner records and cleanup capacity before entry. If later result copying or publication fails, it retains and releases every acquired provider resource through its recorded typed hook exactly once. A null value and an absent ownership record remain different states.

## I preserve logical owners across transport

Opaque scalar tokens already prove issued membership and worker identity; they do not prove provider lifetime. Nested opaque leaves therefore require both the existing issued token and the retained policy's logical pin/drop or provider-lifetime proof. Pointer capture and metadata capacity must be reserved before entry, even if final graph decoding fails. Repeated aliases share an explicitly accounted logical owner; they must not each invoke an independent final drop.

A callback leaf additionally requires issued callback membership, generation, call, path and exact signature agreement. The existing retained V1 format remains limited to its supported shapes. Aggregate callbacks require a named, reviewed successor ABI before either encoder or consumer accepts it. Synchronous callback re-entry on the same busy worker requires an implemented bounded nested protocol; until then I refuse that call before provider entry. A pump alone is not re-entry authority.

VM/COP use one validated envelope definition. COP includes operation/call correlation and bounded nested graph references, retains complete result owners until acknowledgement or abandonment, and drains/revokes callbacks before dropping their code generation. Threaded execution still needs the separately reviewed exec-worker architecture; none of these descriptors makes arbitrary no-exec fork use safe.

## I qualify every consumer before admission

The next source checkpoint must include the exact header/row bytes, generation owner APIs, schema producer, VM and AOT selectors, COP codec, callback ABI decision, all validation/refusal routes and transactional cleanup. An unused retained signature or declaration still receives full validation. Fault controls must cover every measured allocation prefix before entry and every recorded resource after entry; dependency-symbol traps must prove same-image refusal. Original scalar, opaque, callback, visibility, installed read-only/source-hidden and output-failure controls remain unchanged.

I have not implemented or qualified this attachment contract yet. It records the remaining work required to turn the private description generation into executable SDK support.
