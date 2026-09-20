# My service classification within one admission transaction

I retain diagnostics e8ba5f324 and c9fc46d86. At the same measured marker boundaries, 495 full core admissions invoked service-pending classification 8415 times, exactly 17 per entry, accounting for nearly all successful admission time. The unchanged emitter deadline still fails. This checkpoint designs a bounded reduction of duplicate queries, not a longer proof lifetime. I start from qualified 8d73174ec without importing diagnostic macros. Later canonical integration must preserve the subsequently merged ownership changes and receive its own review.

## My lifetime and authority

I compute a service classification on the C stack inside the `!reuse` branch of `vm_core_execute_scoped`. I pass it explicitly through that one synchronous classification call tree. It expires before dispatch; I never store it in VmState, a module, the existing ordinary certificate, global or thread-local state, a returned plan, or a callback. Each fresh core entry that requires classification starts again. Public and nested invocation boundaries, direct public core execution, non-ASSERT traps, false ASSERT, raw memory stores, callback pump and tracing exclusions retain the existing invalidations and checks without change.

The result is the exact existing `nvm_service_execution_pending` Boolean. It does not validate CODE, ownership declarations, layouts, constants, signatures, reference state or frame state, and cannot authorize any target. I retain all those checks and their order. I do not infer safety from module pointer equality alone: equality binds a freshly computed fact to the exact module within this bounded read-only call tree. I make no new concurrent mutation or hostile allocator-interposition guarantee.

## My private interface

I add a private header `src/nanoisa/service_classification_private.h` with the following concrete interface, separate from existing public service declarations:

```c
typedef struct {
    const NvmModule *module;
    bool pending;
} NvmServiceClassification;

NvmServiceClassification nvm_service_classify(const NvmModule *module);
bool nvm_service_pending_classified(const NvmModule *module,
                                    const NvmServiceClassification *facts);
NvmOwnedArrayRoute nvm_owned_array_route_classified(
    const NvmModule *module, const NvmServiceClassification *facts);
bool nvm_mixed_samples_candidate_classified(
    const NvmModule *module, const NvmServiceClassification *facts);
```

The header includes the existing module and owner-route type declarations. The constructor returns exact module identity and the unmodified service query result, without allocating. A null facts pointer or mismatched module calls the original service query afresh; it cannot be interpreted as service absence. These are private implementation interfaces, not authenticated capabilities. Existing public `nvm_owned_array_route` and `nvm_mixed_samples_candidate` delegate to the identical classifier bodies with NULL facts, retaining a fresh service check for every call. Their existing tri-state/Boolean outcomes and short-circuit order remain unchanged. Complete owner-array and mixed admission functions retain their full independent validation and are not converted into fact consumers.

## My VM threading

I factor private VM required/supported/readiness helpers into implementations accepting an optional facts pointer, retaining current helper wrappers with NULL for every existing caller outside the core transaction. The core constructs current-module facts at the point where the first owner classifier would perform its service scan, only when `vm` exists and only in the non-reuse branch. Null-VM behavior stays unchanged. It passes the facts to both initial classifiers, readiness and ownership support. A true pending result continues to produce the same service-first refusal before ownership layout interpretation; I do not replace error messages or statuses.

Ownership support still checks current module, root module and each linked module, in that order with the same short-circuit refusal. When root and current are identical it consumes the current facts but repeats all non-service validation. A different root receives fresh local facts only after the current module passes. Every linked entry receives its own local classification only when reached; I do not allocate or pre-scan a module table, hoist scans before prior refusals, or trust current-module facts for a different module. Repeated linked entries may each be checked. This yields one service scan for the ordinary current-equals-root case instead of 17, and one per separately visited root/linked entry. Owned/mixed full authorities may still perform additional independent service scans.

## My callback and mutation boundary audit

The measured ordinary path calls owner routing, mixed candidate, ownership declaration validation and owned-transfer scanning. The service predicate reads import/service presence then function CODE via `nvm_file_instructions_present` and `isa_code_has_file_instructions`; these contain bounded reads and instruction decoding, no host callbacks or VM dispatch. Owner and mixed candidate bodies read bounded transport cursors, with no host effects. Ownership-contract validation allocates/frees internal decoded layout storage and calls retained-layout/reference checks; it has no user callback, VM execution, or host-service operation. Private complete owned/mixed preparers likewise analyze inputs and release temporary plans without executing bytecode or calling service handlers. Existing frame/reference initialization mutates VM-owned activation state only after classification, not module declarations.

Before source approval I will audit the actual fact-consuming helper chain and its dependency calls again. If a call can synchronously mutate module declarations through a supported callback, I will stop fact propagation before it or recompute afterward; I will not extend the claimed read-only boundary to accommodate it.

## My review and acceptance sequence

I submit this design before production changes, then submit the complete source checkpoint before fixture changes/execution. Fixtures must count the real predicate in all relevant provider translation units, not only VM-local calls: one scan for a standalone ordinary full private entry, zero on an already permitted ASSERT-only reuse, fresh scans at public/direct/nested/non-ASSERT boundaries, and independent scans for different root/linked modules. Existing precise-count assertions will be migrated only to the newly specified counts with their old behavioral/cleanup assertions retained.

I test NULL/mismatched facts fallback, public classifier freshness after benign module metadata changes, pending service metadata and bare File instructions in root and uncalled helper, malformed ownership declarations and refusal status/output/root preservation, all ordinary raw-store invalidations, callback/tracing exclusions and modeled reentry boundaries. Actual callback/FFI and owner-array/mixed/refusal adjacency remains distinct from modeled hooks. The fixture must retain complete authority and no effects before refusal. I use fresh compiler/provider inventories, explicit true-switch/computed-goto selection, Linux/puck ordinary and supported scoped sanitizer gates, then the unchanged full imported emitter ten-second gate. I preserve first terminals; neither a diagnostic speedup nor focused correctness alone closes that gate or the full roadmap.

## My source checkpoint audit

My implementation keeps the null-module short circuit: I construct facts only when both VM and current module exist. Existing `vm_owned_runtime_ready` and `vm_ownership_supported` callers outside the core delegate with NULL; their private per-module helpers gain explicit facts arguments because they have no outside callers. All public owner/mixed classifier callers remain on NULL wrappers. Complete `nvm_owned_array_admit`, `nvm_mixed_samples_admit` and `nvm_verify_owned_module` bodies are untouched.

I inspected the consuming chain: `vm_module_ownership_required` reads facts, routing, declaration validation and transfer scanning; declaration validation uses `nvm_retained_layouts_valid`, fresh decoded layouts, cursor readers, descriptor/path and reference-place checks, then frees its own layout copy. Retained-layout validation decodes and compares table counts/names, then frees its copy. Reference-place validation allocates/frees only its traversal marks. Those calls write outputs/temporary analysis storage, not module declarations. `vm_owned_runtime_ready_scoped` additionally reads instantiated constant counts/pointers. `vm_ownership_supported_scoped` repeats all non-service checks in current/root/linked order and allocates no persistent fact storage.

The possible full-authority branches remain unchanged analyzers: owner admission uses a shallow local parameter view and copied plan, mixed admission uses copied proof/signature/layout storage, and scalar-owned verification uses structural validation, affine analysis and decoded functions. Their disposal releases internal plan/analysis storage, not VM objects or user finalizers. `vm_mixed_invocation_prepare` copies plan signatures/record identities into its stack result and frees the plan. No consuming path invokes VM dispatch, the callback pump, FFI, a service handler, or a user callback. The facts block ends before frame/reference setup, instruction dispatch, value release, or host-trap processing. Standard allocation failure remains handled by the existing analyzers; allocator interposition that arbitrarily mutates caller module memory is outside the documented guarantee.

I found no supported reentry or module mutation while a fact is consumed. This is a static call-boundary audit, not executed acceptance; complete source review and the planned fixtures/gates remain outstanding.
