# Complete-module map publication query

I record this design against d67d11eba for [#1001](https://github.com/jordanhubbard/nanolang/issues/1001). This is a contract-only checkpoint. I have not implemented the query or changed either publisher.

## Scope and retained boundary

I publish optional exact STRING→INT map-field authority only when the complete staged candidate satisfies both my current VM map proof and my current native target boundary. I do not encode native restrictions into MAP_FIELDS, filter source spelling, or infer eligibility from a declaration alone. Main and independently selected original-shadow modules are separate candidates.

My qualified native intersection includes NUL-free executable literals, immutable transport and exact STR_CONCAT. Other ROOT_STRING producers, CAST_STRING, STRUCT transport and nonidentity record ordinal/layout remain outside that intersection. My VM retains counted-STRING semantics. Unsupported dynamic strings remain a scope gap, not completion of #1001. HM_DELETE, other map pairs and standalone map signatures/globals retain their existing proof boundaries. Optional omission preserves the original module without new facts; it does not establish selected execution for an omitted profile.

## Current call sites and closure

| Site | Current behavior | Proposed boundary |
|---|---|---|
| `src/nanovirt/ordinary_authority.inc`, staged publication | array admission, then verifier, then metadata assignment | complete map candidate query before assignment; old non-map branch unchanged |
| `src_nano/compiler/nanoisa_codegen.nano`, `nisa_finish_ordinary` | complete text candidate through `OrdinaryProof.array_publication` | host query selects map before array from actual candidate metadata |
| `modules/nanoisa/nanoisa.c`, `nl_nanoisa_array_publication` | bounded text, unverified assembly, array proof/verify, 1/0/-1 | retain ABI; delegate present maps to new module query and preserve absent behavior |
| `src/nanoisa/nvm2c_map_private.inc`, `nvm2c_map_target` | emitter-local boolean image scan | consume one shared typed, non-emitting target classifier |
| `src/nanoisa/managed_array_shapes.c` | owns map admission, execution snapshots and image proof | own shared classifier and module-taking publication query |

`Makefile.gnu` already links nvm2c, nvm2c_shape and nvm2c_callables into its broad NANOISA object closure. `modules/nanoisa/module.json` includes managed_array_shapes and its decode/heap/proof providers, but not those three emitter providers. I place the non-emitting implementation in the existing managed_array_shapes translation unit rather than adding native code-generation dependencies to the compiler-host module. I will verify both real closures; a broad Make link alone is insufficient evidence.

## Proposed interfaces and ownership

I propose `record_map_publication.h` with:

```c
NvmArrayEligibilityResult nvm_record_map_publication(const NvmModule *candidate);
```

The caller provides a stable, valid C object graph for the duration of this const query. The query returns no plan, image, execution token or generated source. It performs fresh `nvm_record_map_admit`, independently copies/reproves via `nvm_record_map_image_copy`, frees the source plan after the copy terminal, runs the exact shared native classifier against the image, and frees the image on every terminal. It does not call public native emission or re-enter public verification from the internal proof path.

I propose `record_map_native_target_internal.h` for the shared image classifier. Its typed result contains eligibility status, the existing native diagnostic (including an empty diagnostic where the old helper returned false without text), and logical scan work. This is an explicitly internal immutable-image interface, not an execution or source-generation capability. Its implementation lives in a new `record_map_native_target.inc` included by managed_array_shapes. A separate `record_map_publication.inc` owns the module query. nvm2c retains its diagnostic fallback and generation path; only its target scan moves. I preserve public/private native root distinctions.

No Analysis, VmState, VmHeap or old plan layout changes are needed. I do not use padding, global registries or a retained mutable query budget.

## Classification and ordering

I distinguish malformed data from unsupported native domains:

- ELIGIBLE: both independent proof stages and exact native target checks succeed.
- UNRESOLVED: a well-formed candidate is outside the proof profile or an explicit native domain restriction (including an actual embedded NUL, unsupported producer/STRUCT, or valid nonidentity ordinal/layout).
- INVALID: malformed copied recipe, operand width/bits, function/range/count, descriptor/layout shape mismatch, missing string storage, or malformed map metadata. A missing string is not equivalent to an embedded NUL.
- LIMIT: an existing proof limit or the explicit publication scan allowance is exhausted.
- MEMORY: an owning stage cannot allocate.

I validate correspondence across the complete copied image before returning an unsupported-domain result; an earlier unsupported opcode must not hide later malformed rows. I retain the first native-domain diagnostic for valid unsupported images. Successful native outputs and current diagnostics for existing well-formed refusals remain byte-identical. Whitebox corruption may receive a more precise typed INVALID result without becoming optional omission.

Map presence precedes array selection, including mixed map/array metadata. ABSENT preserves current non-map handling and diagnostics. INVALID or observable malformed map identity never falls back. DEFER retains its existing full-old-validator contract; it cannot become a new map omission or legacy-execution shortcut. The public module query itself does not claim absent candidates are publishable.

At publisher hooks I map ELIGIBLE to publish, UNRESOLVED/LIMIT to omit all newly staged optional facts, and INVALID/MEMORY to hard failure. I retain existing 1/0/-1 host semantics and original source/output sentinels. Staging remains unpublished until the full decision and existing final verifier guard succeed. No partial layout/ownership assignment is permitted. Final verification is an additional existing guard, not part of the query accounting below.

## Bounded work and live ownership

I do not claim one shared existing proof budget. The first admission and copied-image reconstruction are sequential independent stages, each with its existing 128 MiB execution reservation and 33,554,432 logical-work cap. The image constructor already measures retained source-plan payload and reserves that live overlap, the image owner, destination proof/snapshot and copied rows under its unchanged cap. I must not add two entire 128 MiB reservations as if both transient proofs remain live.

If A is the first stage peak and I the image constructor overlap-inclusive peak, the query's owned heap peak is bounded by max(A,I), plus only any explicitly measured new owner allocation. The proposed classifier/query add no heap owner. The caller's original module and compiler staging are borrowed external storage and are reported separately. The first stage's transient analysis is destroyed before the image stage. After copying, the source plan is freed before target scanning; the image alone remains.

I add an allocation-free target scan counter with a publication allowance of 33,554,432 logical units. Before work I charge 64 for header/entry/initializer checks, 32 per descriptor, 16 per function, and 128 per instruction plus its decoded byte width. These conservative fixed charges cover direct indexed copied getters and bounded operand comparison; literal inspection additionally charges its exact counted length before memchr, once per executable PUSH_STR occurrence, including repeated indexes. Every addition and range is checked before scanning; unknown widths and inconsistent counts are INVALID. This is logical work, not allocator or machine-instruction timing. ISA decode has bounded instruction width; I will assert/audit that bound before relying on the 128 charge in implementation.

The query's total logical allowance is at most three times 33,554,432 units, comprising the two existing stage allowances and target scan allowance, not one reused cap. Native emission uses the same classifier with its existing proof-bounded traversal policy and no new publication-only refusal; its counter allowance can be UINT64_MAX with checked arithmetic, while publication passes the explicit smaller allowance. This keeps policy shared and avoids changing native diagnostics or successful bytes merely by extracting the scan. Exact tests must measure stage work/peak separately rather than label their sum a simultaneous allocation.

## Implementation and qualification order

1. Extract the target scan with typed invalid/unsupported separation, exact existing native diagnostics and no policy widening. Add the module-taking query beside existing proof/image owners; no publisher edits in this first component.
2. Qualify direct query against actual literals/concat candidates and original-shadow candidates; unsupported string/STRUCT/nonidentity negatives; malformed operands/recipes/shape/late corruption; exact scan cap and one-below refusal/recovery; every allocation prefix and source preservation. Require no C output allocation or emitter object dependency.
3. Run actual native/public-private neighbors and byte goldens, both host closures, strict compilers, six old proof budgets and old map plan/image counts. Audit exported symbols: only module query public, internal classifier no trust capability, observation APIs unchanged.
4. After separate review, wire complete staged C/Nano publisher candidates, retain their existing final verification and atomic outputs, and qualify paired producers/program-shadow independence. Do not change expected source publication before consumer/query evidence is green.

My prospective production files are the three new headers/includes above, `managed_array_shapes.c`, `nvm2c_map_private.inc` and exact Make include dependencies. Later publisher wiring is separately reviewed in ordinary_authority.inc and the existing host facade; module.json must remain free of emitter providers. Test files/required target registration will be proposed with that source checkpoint. Current work changes only this contract and ROADMAP.
