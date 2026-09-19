# My exact owned ARRAY public activation

I track this checkpoint as `task_01e144c13aad46faa8d05d6384270649` under430220. I depend on canonical private runtime/accounting PR836 and its frozen431a qualification. This document grants no execution permission by itself. Paired source18731 and mutation/builtin-identity prerequisites remain separate required work.

## I select one authority before execution

My existing `nvm_mixed_samples_candidate` is deliberately broad: any RESOURCE declaration plus any ARRAY field selects it. It therefore also catches owner ARRAY fields, although its prepared Samples authority requires ordinary managed records. I must split routing before adding public admission; trying Samples then falling back after failure would be wrong.

I add an allocation-free, bounded internal routing query with three outcomes: not selected, owner ARRAY selected, invalid descriptor envelope. It scans original ownership flags and layout field rows using checked cursors, the existing version/count/field bounds and exact table alignment. A directly ARRAY-tagged field in a RESOURCE layout selects owner ARRAY. Nested prior owner layouts retain original identity: their own ARRAY field selects the same module. Ordinary Samples ARRAY fields alone do not select this branch. A module containing both ordinary record fields and owner ARRAY fields selects owner ARRAY and must pass its full unchanged authority; I do not infer permission for their composition.

The routing query is not an element-type, lifetime or structural proof. Invalid descriptor envelopes are refused. Selected modules always freshly call a distinct executable admission wrapper around `nvm_prepare_owned_array_authority`; failure is terminal at that entry, never a retry through Samples/scalar/managed profiles. Nonselected existing profiles retain their own guards. Missing or malformed metadata cannot acquire an ordinary fallback that previously refused it.

Service presence has first priority, before this classifier. Shared execution and translation keep service refusal. The converter's already reviewed private service transport remains first and unchanged; it does not become executable authority. I do not modify File service lifetime/provider contracts.

## My admission wrapper owns runtime policy

The internal wrapper returns an owned opaque plan only after fresh origin, independent lifetime/scalar, common structure and metadata preparation completes. Its call graph is public entry → selected wrapper → private prepare → internal `verify_structure_checked`, never public verifier recursion. No caller supplies a trusted boolean, plan or certificate. Origin-only PROVED remains insufficient.

I retain all existing numerical bounds, opcode refusals, original RESOURCE/layout/path maps and exact signature/local facts. I check function0 as the sole no-argument hosted entry with one INT/BOOL/U8 result, main header entry0 and no `__init__` function. All functions, including uncalled helpers, must pass full authority. Exact function arity/local/result tables, local binding/debug metadata and computed max-stack remain checked rather than inferred from scalar defaults. Owned returns and internal calls keep exact nominal layouts. FLOAT owner fields, ordinary-record composition, borrowed roots, external calls, globals, closures, callbacks and unmodeled opcodes remain refused as established by the authority/runtime contracts.

Public verification does not mutate the module or cache permission. Failed getters and max-stack queries preserve caller outputs. Runtime preparation and emission independently revalidate current immutable module bytes; a prior verifier success is never authority for a later modified module.

## My finite public boundary inventory

Line references below describe reviewed4a5e source and may move during implementation.

| Boundary | Required action |
|---|---|
| verifier.c `verify_function_impl` (515), `nvm_verify` (1041), `nvm_verify_linked` (1067), max-stack wrapper | Service first; owner routing before Samples; fresh full admission; exact queried function max-stack only on success. Zero linked modules may use the profile; any nonzero linked graph containing it refuses. |
| verifier.c closed-profile dispatch (1128) and LLVM/Wasm/HL callers | Explicitly retain owner ARRAY refusal. A successful general verifier cannot silently widen a closed backend. |
| vm.c ownership-required/supported (211/221), invocation admission (314) | Owner ARRAY requires standalone complete authority, never ordinary checked-handler fallback; construct the distinct owner_arrays invocation proof from fresh exact signatures. |
| vm.c `vm_execute`, `vm_call_function_scoped`, `vm_invoke`, `vm_invoke_callable` (4785/4878/4950) | Only idle standalone root function0 with zero arguments, complete constants and no active references/links/callbacks/tracing. Preserve the existing result conventions and output sentinels. Direct helper entry and closure/capture entry refuse before consuming caller values. Internal calls retain the qualified synchronous proof. |
| vm.c `vm_core_execute_scoped` (1367) | Internal synchronous execution consumes only the fresh distinct proof. Actual optional tags and private-qualified retention checks become active for the publicly selected profile in both dispatch forms. PRINT traps retain this invocation's proof until the synchronous caller handles them. |
| vm.c public `vm_core_execute` (4476) | I do not newly admit caller-constructed owner ARRAY continuations. Without an internal invocation proof I return checked refusal before executing a handler; this low-level API cannot substitute forged frames for a started root invocation. Existing Samples continuation behavior stays unchanged. Failed synchronous owner invocations drain their actual stack/frame/callable/reference roots through the qualified scoped cleanup. |
| nvm_v2_convert.c `conversion_ownership` (51), conversion selection/depths (101/209), inverse conversion | Service transport first. Selected owner ARRAY transport freshly prepares full authority and uses its exact function depths, original layout/ownership/path bytes and maps. No descriptor rewriting or fixed scalar max-stack fallback. Existing converter result/storage lifetime and failure cleanup remain intact; refusal never returns a successful partial module. |
| nvm2c.c entry (5920), nvm2c_owned.h shared emitter | Service first; select exactly one fresh authority; reuse the qualified owner_arrays carrier with zero ordinary-record map. Stage full generated C before return; failures return no generated text and CLI output remains preserved. No caller-supplied plan API. |
| borrow_codegen.inc final producer verification (989) | Inventory only in this slice. Its existing Samples/scalar distinction must not be bypassed. Paired source18731 will add the explicit owner ARRAY source profile and fresh final authority after this activation is qualified. |

I audit aliases/wrappers of these APIs, link-table publication and decoder rebuilding as part of implementation review. Every public path either reaches the fresh wrapper before side effects or has an explicit checked refusal. I keep the private test-only entry adapters for regression qualification without making their symbols part of the normal ABI.

## Ownership and failures remain exact

No runtime carrier changes are intended: category1 unique shells, category2 mutable ARRAY handles and separately retained STRING values retain the private runtime protocol. Failed optional comparison checks precede payload access; failed allocation/growth preserves promised state, visible prefix effects and output sentinels. Actual stack/local/pending/callable roots are drained before managed context disposal. The growth-accounting correction and exact pre-GC graph audit remain qualified prerequisites, not a reason to relax cleanup assertions.

Whole-owner references and reference-bearing root metadata remain refused. Nonzero module links, callbacks, imports requiring external invocation, `__init__`, active host frames and direct helper entry remain outside this profile. No public fallback executes partially accepted modules after any selected failure. Permission is scoped to one synchronous root invocation, not persisted in the VmState or module.

## Ordered qualification

1. I review and merge the private runtime prerequisite before activation qualification. I freeze a canonical integration pin and actual tool/source inventories.
2. I implement classifier/admission plus all listed delegates, then request full production review before new public fixtures execute. Provider/link closure and normal symbol surface are part of this checkpoint.
3. Fresh ordinary modules from the reviewed seven-case builder exercise public verifier/per-function/max-stack/zero-link APIs, raw/v2 conversion and byte equality of original retained maps, normal VM execution/invoke/callable entry and normal native CLI output. Both VM dispatch forms and fused/unfused observations retain exact statuses, integer observations and root cleanup. Native GCC/Clang O0/O2 and scoped sanitizers retain allocation-prefix failure/recovery controls.
4. Refusal controls cover selected-but-incomplete authority, wrong entry/signatures/metadata, unconsumed owner, borrowed roots, optional-FLOAT misuse, ordinary-owner ARRAY composition, service precedence, nonzero links/helper/closure/direct-core entry, and all closed backends. Refused modules are never executed; caller outputs and existing destination files remain unchanged where the public API promises publication atomicity. Allocation failures cannot select another profile.
5. I run existing scalar/owned nested/STRING/Samples/File provider adjacency. I preserve first terminals, immutable tools and exact instrumentation scope. No source acceptance is inferred from assembled module success.
6. After canonical activation, source18731 and mutation/builtin-identity tasks qualify the unchanged original Bundle/PREFIX/shadows and mutable aliases. Parent430220 remains open until those required obligations and platform acceptance are met.

## Concrete transport checkpoint

I distinguish loader confirmation from the public converter itself. Existing modules/nanoisa/nanoisa.c checks nonzero v2 max-stack declarations after conversion; nvm_v2_to_nvm_module currently drops that field. My new owner ARRAY converter branch must also compare each nonzero declaration against its fresh exact computed depth before publication. Zero keeps the established undeclared convention. This is limited to newly selected owner ARRAY transport; existing profiles and converter zero/init output conventions remain unchanged. The allocation-free route preserves the no-RESOURCE fast path, so ordinary advisory metadata does not acquire new owner-layout validation.
