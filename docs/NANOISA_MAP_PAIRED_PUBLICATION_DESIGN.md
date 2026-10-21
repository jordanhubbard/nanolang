# Paired STRING→INT map-field publication

I base this contract on integrated ca00599d2 and the qualified [complete-module query](NANOISA_MAP_PUBLICATION_QUERY_DESIGN.md). This checkpoint changes documentation only. I have not changed a publisher or rebuilt a frontend.

## My exact first source slice

I publish checked ordinary record fields whose declared pair is exactly STRING→INT. I reuse kind5/revision1 without a wire-format extension. My initial anchors are unchanged `tests/nanoisa/fixtures/map_record_fields.nano`, the separately retained one-expression concat variant, and `map_nested_u8_borrowed.nano`. I qualify actual program and selected-shadow candidates independently. Other pairs, map-valued fields inside map values, standalone map parameter/result/global obligations and HM_DELETE retain their existing unsupported boundaries. Record-bearing globals remain subject to the complete existing proof and native target query.

I do not filter executable source syntax or claim that every STRING producer has native parity. The complete candidate query may omit optional facts for unsupported/unestablished proof or native domains, including current unsupported dynamic STRING, STRUCT and nonidentity layout cases. The original base module still follows its unchanged validating compiler/output path. A successful omission is not new selected execution authority or completion of #1001.

## Checked declaration binding

In C, `publish_ordinary_authority` retains `ordinary_declarations`: registered declaration pointer/ordinal identity, complete imported source coverage, duplicate/ambiguous-name refusal, ordinary nonresource/nonextern records, and acyclic record graph. I add one exact map branch before the current blanket type_param_count rejection. It requires checked `TypeInfo.base_type == TYPE_HASHMAP`, the field's coarse declared type agreeing, the builtin HashMap generic representation, exactly two non-NULL type parameters, no array element or service identity on the map, and concrete scalar STRING then INT leaves. Leaves have no element, generic parameters, service identity, tuple/function/row or type-variable payload. Malformed/missing or other generic type information omits optional facts; I never recover the pair from HM_NEW or a field name. The parser currently stores the builtin generic_name and two TypeInfo arguments; I preserve that representation rather than weakening generic-record rejection globally.

`ordinary_slot_tag` currently maps HASHMAP to TAG_COUNT, so valid local/staging facts must acquire the coarse TAG_HASHMAP tag. That tag alone grants no exact map identity. If no exact map field binding exists, any newly recognized map local keeps optional facts omitted, preserving the old non-map behavior. Standalone map signatures remain rejected by the map proof. Existing array/scalar/record paths and disabled-name behavior stay unchanged.

In Nano, ASTStruct.field_types contains canonical type strings already checked by the source pipeline. I reuse the exact builtin representation recognized by `nisa_map_value_type`: `HashMap<string,int>` only for publication; STRING→STRING remains a lowering capability, not this authority pair. The field still belongs to the existing exact ASTStruct index/source-owner bindings, including `nisa_bound_ordinary_declarations` for imported candidates. Matching canonical checked type representation is not a filter on executable expressions. I add TAG_HASHMAP (13) to the ordinary tag classifier only for this exact type and track map local/staging slots separately, resetting that bookkeeping with all other ordinary state. A map slot without an exact map field causes omission, just as an array slot without array fields does now.

## Exact wire construction

I preserve every existing layout ordinal, field order, array type row and array binding index. A map field encodes TAG_HASHMAP, zero reserved bytes, nested_idx=NO_INDEX and auxiliary referent=NO_INDEX. The separate MAP_FIELDS table supplies both scalar endpoints.

For M map fields, my payload is exactly `36 + 12*M` bytes:

- u32 scalar_count=2; sorted scalar rows INT then STRING, zero reserved bytes, NO_INDEX referents.
- u32 pair_count=1; pair `(key_type=1,value_type=0)`.
- u32 binding_count=M; rows `(layout:u32, field:u16, reserved:u16=0, pair:u32=0)` in layout/field order.

Every map field appears exactly once. The scalar/pair rows are used and unique. I do not reuse ARRAY_FIELDS row indices for the map pair. Existing ARRAY_FIELDS kind2 rows remain byte-identical, with scalar/nested depth1–3 referents unchanged. Map-only candidates have one extension (kind5); mixed candidates have kind2 then kind5. Both use ownership version3, the existing empty path table and unique ascending extension kinds. I retain simultaneous max4; I neither introduce reserved extension entries nor rewrite unrelated function/parameter metadata.

If B is the existing base ownership length, A the array binding count, T the existing array type count, and M>0, total ownership length is:

`B + 12 + (A ? 8 + (8 + 8*T + 12*A) : 0) + 8 + (36 + 12*M)`.

The 12-byte common suffix is path-table length/data plus extension count; each extension header is eight bytes. Arrays-only stays its exact old `B + 28 + 8*T + 12*A`; no-map/no-array stays version1. I require checked arithmetic before growth/hex construction and assert final writer position equals the allocated length. New map-only accounting preserves the existing 256-layout/65536-total-field and map binding ceilings; source compilers retain their stricter existing record limits. A bounded prepass can count map fields/total fields before new map staging allocations. Unsupported/over-limit map publication omits facts without changing old map-free allocation policy. Source text query cap remains 16 MiB and the existing proof/native caps are unchanged.

## Complete candidate dispatch and existing host ABI

In C, `ordinary_authority.inc` encodes layouts directly for arrays OR maps, since the legacy retain-only branch is not map authority. After all layout/ownership/function rows are staged, map-bearing candidates call `nvm_record_map_publication`; arrays-only retain `nvm_record_array_admit`, other old candidates retain current validation. ELIGIBLE must still pass the existing final verifier before the two buffers are atomically assigned. UNRESOLVED/LIMIT free all new staging and leave original module bytes untouched. INVALID/MEMORY or a failed final verifier report a hard compiler error with no output publication. No partial array-only facts survive omission of a mixed candidate.

In Nano, `nisa_ordinary_publication` builds both extensions, then `nisa_finish_ordinary` submits the full body + facts + existing callable metadata. It must not query a declaration fragment or source program while emitting a different selected-shadow body. Since map facts use version3, the existing full-candidate hook already reaches the host facade. No new extern ABI or builtin arity/result registration is required.

I propose extending the semantics of the existing `nl_nanoisa_array_publication(const char*)` / `OrdinaryProof.array_publication(string)->int` without renaming it. The function already means optional ordinary facts eligibility and returns1/0/-1. Its new dispatch after bounded unverified assembly is: observable map identity first, then the map query; otherwise the existing array/non-array branches. Map INVALID is hard -1, LIMIT is0; PRESENT consumes the query's exact result. ABSENT/DEFER retain the old rejecting full-validator path rather than granting absence. The host keeps freeing its assembled module on every terminal. Query ELIGIBLE plus current final verification maps1; conservative unsupported/limit maps0; invalid/memory maps-1. Existing NULL, oversized input, assembly failure, old array statuses and diagnostics remain unchanged. I can retain the old error text for existing paths and use a map-specific error only where a new map candidate hard-fails.

The host facade needs the map admission/publication headers, but no emitter provider: the exact module.json closure has already linked and executed the shared query without nvm2c. I will verify the real C-hosted module build as a separate approved gate, not infer it from a Make object link. I do not add a caller-supplied plan or cached eligibility token.

## Independence, failure and compatibility controls

I require exact decoded scalar/pair/binding rows and C/Nano byte agreement for map-only and mixed array/map candidates, including field index not zero, multiple map fields, forward record fields, aliases and the existing nested-U8 conversion anchor. Wrong pair/referent/reserved/count/duplicate or missing binding must refuse through actual public transport, not merely a test decoder. Existing non-map scalar/flat/depth2/depth3 byte goldens and all six proof budgets remain exact.

The first unchanged ordinary anchor and independently selected original-shadow input must each publish only after its own complete query. A paired variant with an unsupported operation only in a shadow must retain the ordinary candidate's independent decision and omit the shadow candidate's optional facts. I retain raw verifier/VM/native behavior for omitted candidates, rather than treating a failed query as a source rejection. I include current dynamic-string refusal and DELETE as omission controls, never as a native semantic widening.

I test source output sentinels on hard staged-encoding/query errors and allocation failures. Missing, malformed or unsupported checked type information follows the existing optional-facts omission boundary rather than inventing source rejection; such omission must preserve the original source execution assertions. C staging allocation prefixes cover initial data/layout rows, growth and final encoded layout allocation with exact cleanup. Host assembly/query allocation failures return -1 and preserve file-output sentinels through callers; query LIMIT returns0 with no partial candidate. Nano tests check complete string/base return and nisa_ok/error behavior, then actual source compiler output. I distinguish direct host API injection from producer allocation coverage and do not claim one from the other.

I use the existing source pipeline's shadow requirements, exact source/provider pins, actual public verifier/VM/native transport and O0/O2 forced-GC native controls for published inputs. No declaration-only success substitutes for runtime. Unsupported dynamic strings and other pairs remain explicit open scope after these gates.

## Proposed source ownership and sequence

I propose changes only to `src/nanovirt/codegen.c` ordinary tag bookkeeping, `src/nanovirt/ordinary_authority.inc` exact map binding/wire construction, `src_nano/compiler/nanoisa_codegen.nano` equivalent metadata/bookkeeping, `modules/nanoisa/nanoisa.c` query dispatch and the adjacent host API documentation/shadows where needed. Make dependencies and narrow source/publication tests follow those files. I do not change the schema, proof/runtime/native policies, module.json provider list or old budgets.

1. Review this contract and checked TypeInfo/canonical-type representation before source edits.
2. Draft paired wire encoding + host dispatch, retaining old branches and all source sentinels; freeze for review before a build.
3. Run one bounded current C-seed/host gate on unchanged ordinary plus independent original-shadow candidate; stop on first field/query/consumer discrepancy.
4. After review, qualify paired canonical source generation, omission/error/malformed/OOM controls and old bytes. A fresh compiler/driver build requires explicit worker coordination; installed stages remain pinned baseline providers until rebuilt qualification is authorized.
5. Integrate required gates and retain honest remaining authority/native/hosted boundaries. I make no completion claim from this design.
