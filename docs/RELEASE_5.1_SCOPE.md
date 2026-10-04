# 5.1 Scope - One IR

I use this punch list to distinguish my release requirements from follow-on
experiments. Line references point to the unchecked Phase 20 rows in
`ROADMAP.md`; that roadmap remains the detailed contract and evidence ledger.

## Ownership and services

- **Required (9511):** Replace the C seed ownership prototype with path-sensitive analysis; both frontends need the same affine rules.
- **Required (9538):** Preserve verified ownership facts through the module, linker, reconstruction, and every shipped translator.
- **Required (9547):** Connect generated File and Result descriptors to a verified service-call boundary; integer wrappers are not enough.
- **Required (9549):** Integrate the reviewed five-method File descriptor query before executable service publication.
- **Required (9551):** Qualify private File provenance and Result transfer and cleanup before public admission.
- **Required (9558):** Complete File loops, indirect calls, multi-borrow calls, cleanup, source publication, and selected shadows.
- **Required (9559):** Review matched File and Result flow through VM and native dispatch before changing public selection.
- **Required (9560):** Qualify paired File generation and all shadows through the C seed, Stage 1, and Stage 2 on Linux and Darwin.
- **Required (9561):** Admit File publicly only after verifier, VM, native, generated-binding, and real-file equivalence passes.
- **Required (9563):** Qualify one real local Socket lifecycle; service-handle migration requires a tested socket owner.
- **Required (9564):** Review the private Socket API and close/error policy before host operations.
- **Required (9565):** Qualify Socket lifecycle, refusal, fault, and adjacent File/capability controls on both hosts.
- **Required (9566):** Complete public network/WebSocket bindings and paired verified execution; the local pair alone is insufficient.
- **Deferred-to-later (9567):** Qualify a real CUDA GPU buffer lifecycle; this needs explicit GPU hardware and does not block the portable One IR cut.
- **Deferred-to-later (9568):** Review CUDA driver/context restoration on qualified hardware as part of the GPU lifecycle.
- **Deferred-to-later (9569):** Run real GPU allocation, copy, release, context, and fault gates on qualified hardware.
- **Deferred-to-later (9570):** Extend GPU acceptance to other platforms and public bindings after the portable 5.1 release.
- **Required (9572):** Migrate real File, Socket, and capability handles after ownership and IR enforcement; GPU follows its deferred hardware gate.
- **Required (9575):** Pass one affine matrix across both frontends, NanoISA, NanoVM, and AOT C.

## Compiler product

- **Required (9707):** Infer C-seed loop element metadata for inline and computed arrays; compiler-source lowering depends on valid loops. **Done.**
- **Done (9879):** Propagate transitive foreign header paths through C-seed module compilation.
- **Done (9882):** Reclaim shared static arrays and owned elements at an alias-safe interpreter boundary.
- **Done (9902):** Define Boolean List syntax and backend parity before the compiler product claims list coverage. `List<bool>` now parses and the interpreter, native C and NanoVM backends share its boolean element contract (`tests/nl_types_bool_list.nano`).
- **Required (9941):** Reconcile the legacy `string_from_char` alias so VM and AOT resolve the same host contract.
- **Required (9978):** Prevent ordinary record names from colliding with native runtime helper names.
- **Required (10055):** Carry record and map globals through standalone AOT with correct runtime ownership.
- **Required (10068):** Bind imported globals and selective or qualified aliases in the standalone product.
- **Required (10153):** Make `--emit-nvm` the self-hosted compiler's only backend and make binary output an explicit `nvm2c` plus `cc` pipeline.
- **Required (10260):** Finish the self-hosted NanoISA lowering dual for the full compiler subset.
- **Required (10262):** Compile the compiler to `.nvm` through the C seed and then the self-hosted emitter.
- **Required (10264):** Compare Stage 1 and Stage 2 module bytes; native binary comparison remains a translator test.
- **Required (10266):** Freeze and remove `transpiler.nano` from the product after the compiler module builds itself.
- **Required (10269):** Rename the transpiler compiler phase so the language pipeline ends at NanoISA.

## C AOT, metadata, and profiles

- **Done (10272):** Make C11 the canonical AOT consumer of NanoISA without a VM process dependency.
- **Done (10278):** Cover compiler functions, records, control flow, arrays, strings, modules, and declared host ABI in `nvm2c`; `STR_EQ` is now native (`make test-nvm2c`, 2,431 passed).
- **Required (10280):** Lower `CALL_EXTERN` only through the declared host ABI and refuse unsupported imports.
- **Required (10282):** Keep `wrapper_gen` explicitly packaged-interpreter-only in the CLI and documentation.
- **Required (10400):** Store local names in module metadata for useful reconstruction and diagnostics.
- **Required (10407):** Preserve purity, affine, generic, effect, and exhaustiveness frontend facts in module metadata.
- **Required (10409):** Recover structured `if`, `while`, and `return` from verified control flow for reconstruction.
- **Required (10419):** Define verifier-enforced general and restricted compute profiles.
- **Required (10494):** Resolve the retained evaluator shadow failure before complete compiler-product acceptance.
- **Required (10618):** Finish managed string-array ownership and target execution.
- **Required (10623):** Preserve aggregate and collection identity, mutation, and an explicit cycle policy on the managed runtime.
- **Required (10643):** Establish explicit ordinary heap-bearing record authority before nominal target admission.
- **Required (10648):** Cover array elements, generics, imports, forward order, and remaining nominal record authority.
- **Obsolete (10656):** The shared checked record selection row duplicates completed prior-order and forward-record execution rows; remaining authority is tracked at 10643 and 10648.
- **Required (10661):** Implement exact declared host/module capability linkage and owned results for LLVM and Wasm.
- **Required (10663):** Review the portable read-text linkage proposal as the first concrete portable host capability.
- **Required (10664):** Qualify the read-text query, adapters, opt-in lowering, source publication, and installed links in ordered checkpoints.
- **Required (10669):** Implement the private allowlisted read-text adapters and managed-copy ownership checkpoint.
- **Required (10670):** Qualify native, LLVM, Wasmtime, and Node adapters with real files and denial/fault cleanup.
- **Required (10671):** Extend portable linkage to byte and aggregate results and the remaining compiler capabilities.

## Translators and release gates

- **Required (10672):** Complete LLVM IR translation from NanoISA rather than from the NanoLang AST.
- **Required (10673):** Complete WebAssembly translation from NanoISA rather than from the NanoLang AST.
- **Required (10674):** Cover the applicable language in both translators before exposing them as release targets.
- **Deferred-to-later (10676):** Evaluate JVM, SPIR-V, PTX, OpenCL, and Metal translators; 5.1 does not ship every evaluated optional target.
- **Required (10680):** Run one pinned module corpus through NanoVM, C AOT, LLVM, and Wasm and compare semantics.
- **Required (10684):** Verify that `src_nano` emits `.nvm` as its only compiler product.
- **Required (10685):** Verify that `nvm2c` products do not link `nano_vm`.
- **Required (10686):** Verify byte-identical Stage 1 and Stage 2 `.nvm` products at the release revision.
- **Required (10687):** Verify the pinned NanoVM and C AOT equivalence suite at the release revision.
- **Required (10688):** Verify that `transpiler.nano` is absent from the product compiler.

## Release closeout

- **Required (1033):** Reconcile completed roadmap rows with merged commits and authoritative MAC task states.
- **Required:** Keep main CI green and run the clean release/platform gates on the exact candidate revision.
- **Required:** Draft `RELEASE_5.1.md` after the required implementation rows close, using `RELEASE_5.0.md` as the format.
- **Deferred-to-later:** Tagging and publishing `v5.1.0` remain operator actions.
