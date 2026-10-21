# Public map consumer and transport activation contract

Status: proposed, not implementation authorization. Base db59bc247. Issue #1001 ownership: owned_calls on design/5.1-map-public-activation. This phase changes no publisher and claims no completed #1001 scope. The private borrowed invocation, import, proof, copied image and native receipts establish prerequisites, not public entrypoint qualification.

## Authority and target eligibility remain separate

MAP_FIELDS kind5 revision1 declares exact key/value referents and nominal field bindings. The first executable pair remains STRING→INT with HM_NEW/SET/HAS/GET/LEN, never DELETE. All existing proof ceilings, same64 origins, historical writes and GET INT|VOID remain. Neither HAS nor a literal spelling proves presence. Wellformed other pairs remain executable refusal.

The smallest paired source evidence is unchanged tests/nanoisa/fixtures/map_record_fields.nano from seed, C-hosted Nano and both retained stages. Source-global and nested-U8 compositions are additional retained VM evidence; their raw original modules remain separate from explicitly annotated consumer inputs. Publication of authority by either source producer is still pending.

VM STRING keys are counted bytes, including NUL and admitted dynamic producers. Native nvm2c_map_runtime.inc uses strlen/strcmp and hash-until-NUL. Its qualified target is narrower: NUL-free executable PUSH_STR followed by proved transport; no ROOT_STRING recipe, CAST_STRING, unaudited STRING producer, STRUCT transport or nonidentity ordinal/layout. Preserve nvm2c_map_target exactly, including unused descriptors and unreachable instructions. Metadata names are not executable literals unless consumed. Do not encode these target limitations as pair authority or filter publisher output by favorable syntax.

Public native translation may refuse a VM-valid module atomically with an exact target diagnostic. Broad counted-string/dynamic-producer and STRUCT/nonidentity native parity remain required follow-on where accepted-source evidence establishes the need. If paired publication requires every published module to translate, that native work is a prerequisite, not a waiver. This phase makes no such universal parity claim.

## Classification and selection ordering

Promote a bounded internal framing classifier derived from ownership_map_declarations.inc map_route, with explicit ABSENT/PRESENT/INVALID/LIMIT outcomes. It grants no authority. Metadata-free legacy modules remain ABSENT; a map field without its complete kind5 binding is selected and refused by full declaration/proof, not legacy fallback. Malformed/truncated prefixes, duplicate/unknown frames, bad reserved bytes and framing limits must never become ABSENT.

At every dispatcher, classify before overlapping record-array or ordinary fallback. ABSENT follows the exact old path; PRESENT invokes fresh complete map proof; INVALID/LIMIT stop. Complete framing is checked even after kind5 is seen. Existing unrelated service/owned/linked modes must not be bypassed by map presence: unsupported compositions refuse. Retain maximum four simultaneous extensions; no new wire revision, fixed extension storage growth or hidden trusted flag.

## Exact hook map and proposed order

| Existing file/function | Required consumer-only change |
| --- | --- |
| ownership_contracts.c extension_suffix_read_map / extension framing; ownership_map_declarations.inc map_route | Internal classifier plus explicit map-aware validation route. Keep old NULL-view validators and declaration-only status distinct from execution. |
| verifier.c nvm_verify, nvm_verify_function, nvm_verify_function_max_stack, nvm_verify_declared_max_stacks | Select full module map proof first, then return copied per-function maximum stack/signature results with index/output checks. A function query cannot validate only the requested body. |
| verifier.c nvm_verify_linked / nvm_verify_profile | Explicit refusal for unqualified linked or backend profiles, including map-bearing linked children. No legacy fallback after selected proof failure. |
| nvm_v2_convert.c conversion_ownership | Fresh complete execution proof; extract every required signature/maximum stack before freeing its owner. Set ordinary selected semantics, not owned-transfer authority. |
| nvm_v2_convert.c nvm_private_from_module (public nvm_v2_from_nvm_module) | Include selected map in checked-profile handling and exact map-enabled layout decode. Stage output; failed proof, layout or stack checks free temporary output. Preserve all old NULL-service paths. |
| nvm_v2_convert.c nvm_private_to_module (public nvm_v2_to_nvm_module) | Retain staged exact layouts/ownership, then independently reprove the complete decoded module before assigning caller output. Never trust wire signatures/declared stack maxima alone. |
| nvm_v2_module.c nvm_private_deserialize (public nvm_v2_module_deserialize) | Classify bounded borrowed section views to choose structural map layout decoder only. This is not execution authority; reverse conversion must finish proof before executable module publication. |
| nvm_v2_module.c nvm_private_serialize (public nvm_v2_module_serialize) | Preserve full section framing, exact pair bytes and declared identities; no omission of kind5. Validate before touching caller byte output or stage bytes for atomic commit. |
| assembler.c asm_assemble_impl / asm_assemble / asm_assemble_file | Existing final nvm_verify must invoke selected map proof; canonical text must preserve exact executable constants, layouts and ownership. asm_assemble_unverified remains explicitly unverified and cannot grant admission. |
| nvm_format.c nvm_serialize / nvm_deserialize | Legacy serializer continues refusing layout/ownership-bearing output; do not erase authority to produce legacy bytes. Versioned CLI/hosted callers must use the v2 conversion chain. |
| vm.c vm_module_ownership_required / vm_module_ownership_supported and public call/resume guards | Recognize selected map separately, enforce standalone/root restrictions, refuse unqualified direct/linked/callback entry paths. Do not turn a verification success into a persistent executable token. |
| vm.c vm_execute | Route selected map to module-taking borrowed invocation before record arrays/legacy; fresh owned preparation and same-plan import remain live through execution. |
| nvm2c.c nvm2c_emit / nvm2c_emit_core; nvm2c_map_private.inc | Public module entry constructs fresh plan and independently reproved copied image, enforces unchanged native target gate, then invokes static core. No caller-supplied image/plan API. |

Implementation must also follow actual CLI/hosted load/save callers through these wrappers and retain their file-publication behavior. A raw wire structure decoder is not a verified executable loader. No save/load route may accept kind5 structurally and subsequently run legacy handlers.

## Ownership, reproof and atomicity

Verification, each conversion of decoded input, each public invocation, and each native translation independently derive their own proof from stable source. No cached success bit, module pointer or private observation query is authority. Within one invocation, retain one owned preparation/boundary plan through fresh capacity-based import validation and execution; do not run a second divergent analysis between validation and use. Module and caches remain exclusively stable throughout the synchronous call.

Borrow the caller VmState; preserve ordinary cycle-aware heap, buffered aliases, ordinal def_idx, globals across calls and explicit result consumption. Root remains zero-arity INT; initializer zero-arity VOID/INT with residue drained. Stack reservation follows qualified rounded growth accounting. Runtime failure uses selected cleanup (stack/frames unwind, globals/committed alias writes survive), not transaction rollback. Preflight failure preserves prior state under the existing preparation contract. No private DAG heap policy or copied-owner/scalar-copy shortcut.

New constructor/getter outputs remain unchanged on failure; buffers/files must be staged until all required checks succeed. Existing APIs that explicitly zero size/error outputs retain that documented convention rather than inventing changed-output preservation. Do not claim destination-file atomicity merely from a NULL C emitter result: qualify the actual CLI save path. Runtime first-error/unwind is distinct from transport output atomicity.

No unconditional old structure growth or budget changes. Map-only owners use existing charged reservations, live overlap and destructor ordering. Old absent-route serialization/native bytes and six proof budgets stay exact. Macro/production availability must be reviewed explicitly: promoting private consumer machinery cannot silently enable unrelated private observation APIs or alter default heap layouts.

## First implementation checkpoints, after review only

1. Internal classifier/common admission and verifier/transport selection, preserving fail-closed execution until the corresponding public VM/native routes are present. Direct API and CLI malformed/refusal tests first; no source publisher changes.
2. Production availability and public VM dispatch to the already-qualified borrowed supervisor, with direct-call/resume/linked exclusions. Test actual vm_execute, never substitute private execution for public success.
3. Public nvm2c selection through fresh copied-image proof and unchanged target restrictions. Test actual public emitter/CLI and exact refusals for counted-NUL, dynamic STRING, STRUCT and nonidentity layouts.
4. Full actual public transport/consumer gate, then separately review publisher interfaces. A partly wired tree must not expose selected metadata to legacy execution.

## Required consumer qualification

Use retained raw producer modules with explicit exact consumer annotations. Preserve original bytes/hashes and identify annotations; do not relabel them source-published authority. Exercise public binary serialize/load, canonical disassemble/assemble, verify and max-stack APIs, metadata/name independence, and public VM repeated invocation with caller result consumption. Test all12 retained VM inputs, source-global aliases, initializer/tail/assert resume, failures and imported buffered state. Native tests use the qualified intersection and exact target refusals outside it, not blanket parity assertions.

Negative transport controls include wrong pair/ref/field, missing and duplicate binding, malformed later frame, five extensions, all count/length/reserved errors, unsupported wellformed pairs, understated stack maxima, historical bad writes, and linked/backend exclusions. Assert no malformed-to-ABSENT fallback. Reproduce GET absence as VOID through public VM/native paths. Verify prior output/file sentinels and allocation-failure cleanup/recovery at actual public boundaries.

Native successful cases require actual O0/O2 sanitizer and forced-GC execution; failure cases require exact expected assertion/allocation termination, never arbitrary nonzero. Old public ordinary/array execution, byte goldens, declarations, codecs and six budgets remain required. No frontend/bootstrap is needed for this contract; publisher builds remain a later reviewed phase.

## Recursion audit and unpublished staging

The selected proof chain is nvm_prepare_record_map_execution → rme_prepare_boundary → rae_snapshot → analyze/analyze_boundary → nvm_record_map_structure_prepare → record_structure_prepare. The structural helper invokes only verifier-internal verify_module_ranges, verify_module_contracts_and_imports and verify_function_body (with a non-NULL budget), not public nvm_verify. Analysis uses structure->stacks; its public nvm_verify and nvm_verify_function_max_stack fallback branches execute only when structure is absent. Map preparation always supplies structure or fails. rae_snapshot copies module storage without public conversion/serialization. Independent image copying calls rme_prepare_reserved on the copied snapshot, not the public verifier. These boundaries must remain explicit: no map proof helper may call a newly selected public verifier/converter recursively. Public conversion/serialization may separately repeat full proof as required by their independent output boundaries; repetition is not recursive authority reuse.

The classifier/verifier/transport source checkpoint is unpublished until all public dispatcher guards and production availability are wired. It must not provide an intermediate public verifier success followed by legacy VM dispatch. Stage selected public additions behind a single internal disabled build gate until the complete consumer routing is reviewed; no target enables that gate during the initial checkpoint. Private APIs and their existing tests continue independently. Enabling the gate is a separate reviewed source change after VM/native guards exist, not an environment-based trust bypass.

Old ABSENT diagnostics and budgets remain unchanged. The routing probe must not reinterpret existing non-map malformed declarations as successful absence, nor impose map-only ceilings on valid old profiles. If no map identity can be established from safely bounded framing, the existing validators retain responsibility and their original diagnostics; a framing error remains rejection, never successful fallback. Once a map field/kind5 is identified, later malformed frames cannot erase selection. Full selected proof remains mandatory.
