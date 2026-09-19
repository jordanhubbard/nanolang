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

I distinguish loader confirmation from the public converter itself. Existing modules/nanoisa/nanoisa.c checks nonzero v2 max-stack declarations after conversion; nvm_v2_to_nvm_module currently drops that field. My new owner ARRAY converter branch must also require each nonzero declaration to be at least its fresh exact computed depth before publication, matching the loader's existing bound comparison. Zero keeps the established undeclared convention. This is limited to newly selected owner ARRAY transport; existing profiles and converter zero/init output conventions remain unchanged. The allocation-free route preserves the no-RESOURCE fast path, so ordinary advisory metadata does not acquire new owner-layout validation.

I also bind supplied function parameter tags to retained authority: a known sidecar tag must equal its exact admitted parameter tag; TAG_VOID remains the established unknown placeholder. My owner ARRAY v2 output writes fresh authority parameter tags even when source sidecars are absent/unknown. This prevents transport from guessing scalar tags or losing exact owner/STRING identities. The private query itself stays descriptive and unchanged.

## My production checkpoint

My actual API is `owned_array_admission.h`: `nvm_owned_array_route` returns NOT_SELECTED/SELECTED/INVALID; `nvm_owned_array_admit` returns existing PREPARED/UNRESOLVED/INVALID/LIMIT/MEMORY with owned output only on success. The unchanged private query does not grant permission. Routing/admission is implemented in `owned_array_admit.inc` inside verifier.c, so internal structural delegation stays inaccessible.

I wire verifier/per-function/depth/linked/closed profiles, both converter directions, native emission and standalone VM entry preparation. The VM copies exact admitted signatures into the already qualified distinct invocation proof and makes the existing operand/retention preflight available to ordinary builds. Direct core execution without that internal proof refuses; direct root function callables are allowed while closure-tagged entry refuses. The converter writes fresh exact parameter tags and depths, and checks nonzero incoming depth bounds without changing its established output-initialization convention.

I leave both source producers untouched. Source18731 must explicitly require SELECTED plus fresh admit after ordinary verification for its new profile; existing Samples/scalar final guards remain unchanged here. Header dependency generation uses the existing compiler `-MMD -MP` mechanism. I integrate canonical File839 providers and true-switch838 qualification before fresh gates, preserving all File closure dependencies. No activation fixtures have executed at this production checkpoint.

## Routing-envelope review correction

Static review ofcac460fee finds my route reads more envelope bytes than it validates. I correct that mismatch before qualification: once any RESOURCE flag selects the resource-bearing scan, all flag bits and RESOURCE→COMPLETE consistency, zero alignment padding and retained function-count word, layout kind/header reserved byte, field tag/reserved bytes and exact layout cursor extent must be checked. These bounded transport checks return INVALID. They do not prove nested nominal identity, per-function declarations, origins, lifetime or runtime permission; full admission still owns those checks. No-RESOURCE tables keep their established ordinary fast path.

## Direct-core refusal preserves caller state

Review off8f7 finds the outer `vm_core_execute` still classifies owner ARRAY as broad Samples and can run Samples cleanup after refusing an unproved continuation. Moreover, the shared `trap_error` helper clears reference activations. Neither operation is appropriate before I admit a synchronous owner ARRAY invocation.

I require the public direct-core wrapper to detect owner SELECTED/INVALID before calling the scoped core or entering Samples cleanup. It returns a type-error trap and may update only `last_error`/`error_msg`; stack values/count, frames/count/owned callable roots, active reference contexts/generations, effect state, instruction/function/module/activation state, heap references/accounting and output remain unchanged. The caller still owns that continuation and its cleanup. I use the existing diagnostic-only `vm_error` path rather than `trap_error`. Existing Samples cleanup and failed admitted synchronous owner cleanup remain unchanged.

Focused controls will retain actual caller-owned STRING/ARRAY roots and snapshot reference/frame/control state around direct-core refusal, compare exact state and counts, then release those roots through the caller's normal cleanup. Refused continuations never enter a bytecode handler. This does not introduce public owner continuation/resume support.

## Fixture expectation migration before execution

Static fixture audit finds only two authority-query assertions and the seven-case private runtime main still require general verification refusal for these newly admitted complete modules. I migrate those exact positives to public verification success while keeping the old shared scalar ownership validator refusal. The runtime fixture will require normal native emission and compare it with the private emitter's output; its former host-root refusal becomes an actual helper-entry refusal with unchanged output/root checks. All unsupported authority cases and private-symbol absence checks remain.

A public test mode reuses the same finite seven-case builder and independent cleanup graph, invoking normal `vm_invoke`, `vm_execute`, `vm_call_function` and direct-root `vm_invoke_callable`. It records the existing `vm_invoke` post-admission VOID failure result separately from pre-admission unchanged outputs and the other APIs' result conventions. Both dispatch forms have explicit macro evidence. Additional controls cover fresh revalidation, converter tags/depths, route envelopes, service/link/helper/closure/closed-backend refusal and exact direct-core state preservation. No fixture has run at this migration checkpoint.

### Checked parameter preparation before public authority

I retain my absent/VOID sidecar promise after the frozen964c fixture exposed my
shared exact descriptor validator's refusal. I do not relax that validator or
change the private query. My public admission wrapper instead prepares a bounded
shallow module view with owned temporary parameter rows before invoking fresh
complete authority.

I first retain service priority and exact candidate selection. I read the
ownership table with my checked cursor: supported version, bounded layout flags,
alignment, exact function count, each declared local/parameter count equal to the
function header, and every eight-byte descriptor within bounds. I cap functions
at eight, parameters at eight per function, and locals at 256, matching authority.
I copy only parameter tags from these declarations into the temporary rows. Each
parameter tag must be a modeled exact non-VOID signature tag; known supplied tags
must match it, and only absent rows or TAG_VOID placeholders can be completed.
I do not infer nominal layout, modes, origin, lifetime or scalar permission from
this pass. My unchanged complete authority on the prepared view must validate
all of those facts, including exact descriptors, reserved bytes, trailing path
data and every function body. A successful cursor pass alone grants nothing.

I use fixed bounded temporary storage for eight row pointers and 64 parameter
tags, initialized before constructing the view. I reject larger counts before
indexing. The view copies the NvmModule header by value and changes only its
parameter-row pointer. Caller-owned module bytes, row pointers and tags remain
unchanged. I retain the original module's code, constants, maps and declarations;
I neither move nor free them. The unchanged private preparation copies the facts
owned by the returned plan, so no returned plan may retain a pointer into this
stack view. I audit that lifetime before implementation. No new allocation is
needed for sidecar preparation; existing authority allocation failures preserve
*out and free partial facts as before.

After complete authority succeeds, my existing hosted-entry and signature checks
still apply. Public transport writes the exact plan tags. Direct private queries
continue to report their existing refusal on missing required sidecars. Other
profiles and shared descriptor validation remain unchanged.

I qualify whole-table absence, individual missing rows, VOID placeholders, known
mismatches, excessive counts, invalid descriptor tags and truncated declarations.
I require unchanged caller bytes/pointers and output on refusal, plan independence
after input destruction, later valid recovery and existing allocation-prefix
controls. I retain the original terminal and run fresh fixtures only after this
production correction is independently reviewed.
