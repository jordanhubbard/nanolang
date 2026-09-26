# I assemble one complete 5.1 candidate

I begin in isolated branch `integrate/nanolang-5.1-candidate`, preserving both original lanes. My first merge joins SDK `fcda5c5231445eb741df338b0e15c4a15084ce4b` and native `4855b75cc4c72f7e734d16d285e4f36cccf557c6`; their common ancestor is `8028ab1949dc40d039c46d5c3d475f21905f82bc`.

I resolve two production files without dropping either contract. In `src/main.c` I retain the SDK opt-in timing include and parent-after-wait observation alongside the shared completion protocol. The resulting file is byte-identical to the SDK parent. In `src_nano/transpiler.nano` I retain the SDK exact `ASTServiceDecl` header/no-local-provider shadow after the native generic forward-declaration checks. Both declarations and the actual native generation changes remain present. The two documentation conflicts retain both histories; their passing and failing results continue to name their original source pins.

My automatic merge also carries the native selected-union projection and constructor spelling, complete selected-payload annotations, local declaration staging, signed length and exact declared-extern builtin authority. The SDK owner/visibility, opaque/tuple carriers, provider/install closure and STRING field adapter remain. Clean textual merging is not semantic qualification: owners must review the combined source before execution. No production or fixture was run for this merge checkpoint; `git diff --check` passes.

I leave my SDK input inventory and generated inventory unchanged at this intermediate checkpoint. After root byte/literal work, backend execution, CI and qualified metadata ownership prerequisites, and required PR522 canonical routing are integrated, I regenerate the inventory from the complete source and review its exact dependency closure. PR949's independent Darwin failure evidence remains a contribution to preserve, not a production fix.

My later source integrations and full both-host C-seed/S1/S2, native/STRING/visibility, original unit04, complete CI/backend, installed source-hidden/read-only SDK, canonical fixed-point and release gates remain required. I do not turn this first merge into a narrower release candidate or relabel previous branch results as integrated acceptance.

## I combine root byte and literal ordering with the retained native contracts

My second merge joins root `3a7665809` to first candidate `69c3aea16`. The intervening native builtin selection work makes nine production conflicts, rather than the seven in the earlier SDK/root-only preview. I resolve them as follows:

- My generic one-argument `env_function_is_builtin` remains the identity authority for native array views. My additive two-argument `env_function_is_named_builtin` first requires that actual initialized registry identity, then compares the exact requested registry row with bounded indexing. Root byte/pop/evaluator/bytecode callers use the named query. Existing lexical/local/extern selection precedes both; I retain the actual selected function identity before argument evaluation. Additive source controls cover correct and incorrect names, null names, copies and externs.
- I expose root's `checked_array_record_name` API through the existing complete owned `NominalView` and registered canonical declaration key. I do not restore root's older short-name/empty-receiver inference. The owner-aware array, union and callable checks, pre-RHS field/local snapshots and failed-publication rollback remain. Root scalar destination comparison runs alongside those nominal checks at effects, arguments, lets, field/local stores and returns. My corrected STRING field adapter remains in that path.
- I retain the native complete tuple typedef and once-only ordered child snapshots; this already preserves root tuple source order without reinstating a flat fallback. Root record/union/scalar-array ordered initialization is added. The Nano union emitter combines root sequential typed-field assignment with native's resolved `actual_union_name` for field context. Global initializer collection stays before derived/prototype emission; root's callback-global shadow and the native list-worklist assertions are both retained, without duplicate global collection.
- The Nano checker keeps native complete branch/tuple/array context and STRING comparisons. Root's explicit byte-array literal conversion runs before the broader array context path. Pop and byte arithmetic remain additive. Native's earlier checked array-operation dispatch remains authoritative for push mutation identity and complete element facts.
- Make rules retain all SDK/native and root targets/dependencies. The NanoISA provider list adds capture bindings while retaining the existing single `nsi_file_plan.c` entry, not a duplicate copy. SDK provider ownership, installed discovery and ABI inputs remain unchanged pending final inventory regeneration.

My strict C syntax-only check first finds a missing `AST_SERVICE_DECL` case in the native private-name traversal introduced by combining these AST domains. I retain that diagnostic and explicitly classify this descriptor-only declaration with the nonbinding leaves; its fields contain no AST expression or source local binding. The corrected exact command `gcc -std=c99 -D_GNU_SOURCE -Isrc -Wall -Wextra -Werror -fsyntax-only src/env.c src/eval.c src/typechecker.c src/transpiler.c src/nanovirt/codegen.c` passes. This checks C syntax and warnings only: no compiler product, Nano shadow, imported snapshot consumer or full gate is executed. Owner review and all subsequent integrations remain required.

## I retain backend execution before CI composition

I merge backend `57542f703` after the root checkpoint. Its only production conflict is an additive Makefile tail: I keep both the root LLVM scalar-tail gate and the private generated-record-array C/LLVM gates with their exact object/flag selection. I retain both roadmap histories and the additional committed-report integrity paragraph from the candidate. No product runs at this source checkpoint.

I next merge CI `82c30f5cc` with its original partition manifest and fixture corrections. The verifier conflicts combine invocation-local `NvmServiceClassification` reuse with the existing capture-binding admission refusals: I keep the capture guard at the same boundary before classified routing, and pass the invocation facts through the classified helpers. Both original capture-refusal and mutable-module/service-classification tests remain separate functions with their original assertions and calls. Strict syntax-only compilation of `src/nanoisa/verifier.c` and `tests/nanoisa/test_verifier.c` passes under `-Wall -Wextra -Werror`; no verifier or other product executes.

The remaining diagnosis head is not an ancestor of CI82: its merge base is `7f9dd7b3a`, though the already integrated prerequisite production changes match. I must not replay shared commits as new corrections. Native `bfe4b7bf1` and VM `4f7317344` have byte-identical complete `copy_metadata_struct`, `copy_metadata_owner`, `metadata_array`, `extract_module_metadata` and `free_module_metadata` bodies. I will retain one owning implementation and the candidate's surrounding module/provider lifetimes. The native standalone metadata fixture still has a fork-only fatal-worker loop, while VM4f uses reviewed fresh process images; I must carry the latter supervision contract before executing either integrated fault path. Production helper equality does not qualify a fixture or erase the prior Darwin sanitizer terminal. These metadata and subsequent canonical/peer integrations remain open.

## I integrate the qualified ownership prerequisite once

I merge native `bfe4b7bf1` and diagnosis `4f7317344` with their ancestry retained. The five production metadata helper bodies named above compare byte-for-byte with VM4f after integration; I keep one implementation, all candidate cache-generation/Environment leases, SDK include/object paths and per-declaration ownership. Existing root capture guards and both verifier fixtures survive the overlapping diagnosis merge. Header comments continue to distinguish Environment-borrowed annotations from independently owned metadata snapshots.

I preserve both existing fixture scopes: the native standalone snapshot/complete-tuple control and VM's original lookup/parser/snapshot default suite. I factor VM4f's exact fresh-image worker, decimal argument parser and bounded diagnostic reader into `tests/struct_ownership_worker.h`, used by both. The native snapshot fault loop is replaced with the exact VM4f loop; it enters the same `_snapshot_fault` worker with checked arguments. Its callback/tuple/lifetime assertions and measured two-mode fault/recovery checks remain unchanged. I do not retain the old fork-only fault route or introduce a second production snapshot implementation. Make prerequisites include the shared helper. This adaptation is source-reviewed before its own execution; VM's four passing configurations remain evidence for VM4f, not this combined candidate.

Strict syntax-only checking of the integrated module/env/parser/checker/transpiler/verifier and both ownership fixtures passes. No product executes. Root evidence successor, canonical PR522 and peer evidence still precede final source freeze and SDK inventory regeneration.

## I preserve canonical products and expose the remaining SDK boundary

PR522 makes the default sibling `.nvm`, explicit native output and `--target c` consume the same verified NanoISA pipeline. Its C target already means nvm2c C11 output. I retain that public contract; I do not turn it into a source-native fallback. I retain the complete source-native generation/provider/shadow implementation as an internal entry point for explicit bootstrap and regression qualification. Such qualification does not establish canonical installed support.

My full SDK acceptance still requires the canonical product routes to support the admitted opaque identities, complete tuple/callback carriers and exact native provider/ABI closure under installed-only, hidden-source and read-only conditions. The older source-native success cannot satisfy this requirement. I must audit the actual NanoISA foreign-service/provider representation and installed translator/runtime discovery, then implement the smallest checked adapter that preserves verification. This is an open full 5.1 dependency, not deferred release scope.

The CI conflict retains the newer partition aggregator and its exact worker results; the older monolithic timeout setting cannot replace aggregate result checks. The C driver keeps current cleanup/Environment destruction guards while adopting the renamed lowering diagnostic phase. Artifact fixture compiler selection and existing instrumentation flags remain authoritative. New canonical reconstruction snapshot dispatch is retained alongside the later typed arithmetic/refusal fixtures.

The resolved driver shares the complete retained SDK front end in `compile_program_mode`. Its public `compile_program` wrapper always selects canonical lowering; the separate internal `compile_program_source_native` wrapper rejects bytecode selection and preserves source-native generation, original shadow compilation, provider preparation and cleanup. I keep both introspection forms under distinct collector names. The moved module-facts scanner retains the original token scanner and its new `pure`-modifier controls. My merger retains opaque declarations, complete parser/owner facts and the reviewed split/prefix behavior.

I retain the candidate's complete nominal/list checker and checked array-record query instead of PR522's older short-name helpers. Bytecode keeps lexical upvalue-before-global selection and INT/ENUM byte conversion. PR522's enum-list result admission follows its explicit enum-to-TAG_INT selection and existing tag-neutral array mutation operations; checked declarations still precede intrinsic dispatch. Nano expected-array emission combines computed-byte conversion with contextual nested empty-array emission, with both shadow sets retained.

The canonical SDK gap is not a link-command-only change: `nisa_type_ok` and `nisa_tuple_type_ok` constrain admitted array/tuple shapes; `nisa_register_extern` accepts metadata, artifact functions or the bounded host ABI and rejects arbitrary opaque SDK extern signatures. I must preserve exact owner/kind and provider ABI in a checked external/service descriptor through verification and VM/nvm2c execution, then provide installed discovery and the original acceptance matrix. Existing service descriptors must be audited before designing an adapter; no arbitrary native-symbol bypass is authorized here.

Strict syntax-only compilation of `src/main.c`, `src/typechecker.c` and `src/nanovirt/codegen.c` passes with C99, GNU feature definitions and `-Wall -Wextra -Werror`. The four conflict-resolved Python sources parse. These are source checks only; mandatory Nano shadows, final inventory regeneration and all combined product/installed gates remain open.

## I combine the qualified byte-array carrier with canonical recursion

I merge native `051b87183` (production220, fixture879) after the installed AOT
role prerequisite. I retain the candidate's owner-aware callable/tuple/opaque
return checks and add the byte-literal destination check without replacing those
contexts. Codegen keeps earlier enum narrowing and lexical upvalue precedence;
its new byte-array snapshot/conversion is additive.

The private nvm2c kind numbers already use13 for recursive arrays and14 for
functions, so U8 arrays use the distinct private value15. This is an emission-
local carrier code, not a serialized ISA tag. Global classification retains the
candidate's resolved `stored_kind` and byte-array shape equality; global loads
retain both recursive-array facts and the U8 guard. Exact scalar-byte facts enter
the existing shape carrier, with ordinary array/function cases unchanged. Both
Make targets and all prior fixtures remain. Strict syntax-only checking of the
checker, bytecode emitter, nvm2c and shape implementation passes with warnings as
errors. Native051 evidence remains attributed to its own source; no combined
product executes at this merge checkpoint.

The first80e8 review found that changing the classifier constant was insufficient:
six newly imported emitted-runtime lines still used byte discriminator13. I
correct only those byte-word roots/record fields and tagged len/get/set/push
cases to15. I audit every literal13/15 occurrence in `nvm2c.c` and
`nvm2c_map_roots.inc`: remaining13 denotes AARR in recursive constructors,
child validation and naarr dispatch/root traversal; TAG_MAP in boxed map
construction, truth/equality and map access/rooting; or ASCII carriage return.
The record root's word-array predicate now includes15 and retains its array
storage member; the separate recursive-array root traverses children under13.
All classifier-driven formatted guards and `array_shape_kinds` use UARR15
(symbolic shift15 still fits the existing uint16_t mask). No wire tag changes.

The additive actual VM/nvm2c O0/O2 fixture keeps a nested string array and a byte
array live across allocation churn, mutates/appends each, and checks nested
contents plus exact U8 element tags afterward. It retains all original byte
controls. Strict C/Python syntax checks pass; no old or corrected product is
executed before source review. This does not claim admission of nested U8 array
children beyond the existing recursive carrier contract.

## I preserve staged arguments with synchronous callback snapshots

I merge qualified evaluator62a77f512 after the corrected byte fixture. The
conflict keeps both complete helpers: native nominal record-list dispatch and
synchronous callback descriptor ownership detection/destruction. Argument
execution stays in the candidate's `eval_staged_argument` boundary; I apply the
new callback ownership capture and return cleanup to that same once-evaluated
value. The callback name snapshot, live Symbol-owner scan, result cleanup and
all root total integer operations remain from62a. Strict evaluator syntax passes;
no merged product executes until review. The unchanged full `test-eval` target
must run on the complete candidate metadata/parser closure, not the narrower
isolated evaluator base. Earlier focused62a results retain their own source scope.
