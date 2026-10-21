# Private native map consumer: review checkpoint

Status: contract only. Based on aggregate c71f22d10 (private observation f4fbe9c55). Issue #1001 ownership: owned_calls. No native/public VM/load/save/publisher selector changes are authorized by this document.

## Existing sites and concrete gap

- `nvm2c.c:2162–2238`: HM_NEW constructs MAP with STRING key and INT or STRING value children; SET relates exact child shapes; GET produces VALUE with optional value payload. HAS does not prove subsequent presence. The selected proof remains STRING→INT and excludes DELETE.
- `nvm2c.c:5174–5230`: existing HM emission calls nmap_owned_new/set/owned_get/has/len. `nvm2c_map_runtime.inc:66–82` copies insertion keys, replaces an existing value, and returns zero-kind VOID for absent GET. Native strings use strlen/strcmp/hash-until-NUL, unlike counted VM strings.
- Public `nvm2c_emit` selects old record-array admission at 7609+, frees that plan and sets canonical_records at 7665. Map metadata must not pass this selector or fall back to legacy ownership admission.
- AGG_PACK already stores MAP in r.m and canonical nrec_create storage (5635+); AGG_GET retrieves r.m (5889). Simulation of AGG_SET (2736+) equates exact replacement field shape, but emission (5763+) excludes MAP and has no m-member replacement case. An eventual change must be private-map-only and preserve old output bytes.
- `nvm2c_shape.c:151` already has two MAP children; shape_record_return at 968+ carries nested shape relations through calls/results. This is representation machinery, not authority for nominal map fields.
- `nvm2c_map_roots.inc:59–123` traces tagged map values and record m fields, deduplicates canonical identities, marks owned maps and destroys unmarked maps. Keys are owned inside maps; STRING→INT values introduce no nested managed value edges. Actual forced collection/alias qualification is still required.

## Proposed private interface and ownership

A compile-time-private header exposes only a module-taking entrypoint, provisionally:

    char *nvm2c_emit_record_map_private(const NvmModule *source,
                                      char *error, size_t error_size);

It accepts no trusted plan, context, image, eligibility Boolean or callback. It constructs a fresh execution plan from source, creates the independently reproved copied image using existing reserved-overlap accounting, then frees the source plan. Every inference/emission read uses the image's borrowed module and its owned frozen plan; the image survives until emission has either completed or failed. Returned C text owns its bytes and contains no source/image pointers. A single cleanup path frees temporary inference storage, image and failed output; failure returns NULL. The caller must keep source immutable during synchronous preparation. Destroying source after that stage must not affect translation, tested via an internal whitebox seam rather than an exported trusted token.

Proposed internal extraction separates the unchanged legacy wrapper from a static emitter core. Only the private constructor can supply its validated image to the core. Public calls pass NULL and preserve old routing/output. Exact interface/cleanup diff requires review before execution. Existing image proof reservations remain unchanged; native inference/output allocation is separately measured and retains current native limits, never represented as already covered by image caps.

## Exact field and operation obligations

Before using canonical map storage, each selected nominal layout/field query must yield exactly STRING→INT. Seed/check the corresponding MAP shape with child0 STRING and child1 INT, not merely VK_MAP. Relate pack/get/set, local copies, joins, parameters/results and record globals to that same shape. Check the frozen instruction recipes against complete decoded instructions, including unreachable code; unsupported operations or inconsistent identities fail before output publication. A per-PC field may have multiple receiver origins, so a layout index must not be guessed from the first origin; all possible proved origins must agree on the required field pair.

HM_GET stays INT|VOID throughout shape inference, C tagged storage and comparisons. No literal-key or HAS-based narrowing. Wrong value/tag/receiver, absent GET numeric use and false ASSERT retain existing runtime failure semantics. AGG_SET MAP requires exact checked receiver and m-field transfer without allocation/collection between operand capture and edge publication; alias identity remains canonical. No DELETE, STRING-value map, external input or broad scalar-tag whitelist is added.

## String byte-domain gate

`nvm2c.c:656` rejects executable embedded-NUL constants. That alone is not yet a proof for every accepted string producer. Existing recipe string producers include CAST_STRING, CONCAT, SUBSTR, TRIM, case conversion, REPLACE, FROM_INT/FLOAT and SPLIT. The private proof also transports strings through locals, arrays, records, calls and globals. Native HM runtime uses C-string keys. Private VM counted-NUL success remains independent.

The first proposed native qualification is the unchanged four-original constant/forwarding anchor, with an explicit private target refusal for string-producing opcodes not yet audited. This is a target restriction, not a new declaration schema or VM restriction. Before relaxing that restriction, audit each producer's VM/native byte semantics, numeric formatting, substring bounds and zero-byte generation; imports/externally supplied strings cannot be assumed safe. No full STRING→INT native parity claim follows from the first anchor. If the first originals contain an unaudited producer, retain that refusal and return for review rather than altering the sources.

## Required qualification and remaining boundaries

Use all four unchanged original modules as independent inputs. Retain public verifier/load/native refusal of selected kind5 while the private entrypoint works. Exercise source/plan destruction, malformed copied fields/recipes and allocation-prefix NULL-output preservation. Compare actual private VM execution to generated C O0/O2 with sanitizers, canonical alias identity, same-map record aliases, detached map history, field replacement, shared/swapped arguments, initializer/entry/tail/ASSERT boundaries and missing GET. Force collection across record-held maps and key copies; check actual live allocation balance independently of native net-capacity counters. Allocation failures may abort generated programs; require exact expected termination and reject sanitizer errors, not arbitrary nonzero success.

Old array/native byte goldens and old proof budgets remain exact. Native provider failure cannot be relabeled as proof refusal. Public VM integration remains a separate lifecycle/result/status contract: the private owner owns copied image/result/globals and cannot simply be installed into a borrowed public VmState. Public selector, loader, serializer and publisher admission await jointly reviewed qualification.

## Reviewed initial target restrictions

The initial executable STRING producer whitelist is PUSH_STR only, followed by already-proved transport (loads/stores/dup/swap/calls/returns, aggregate/array extraction). Refuse every ROOT_STRING recipe, CAST_STRING, character-to-string conversion, reflective/name extraction, imports and any unaudited string-producing instruction before native core entry. Metadata names are not executable strings unless an instruction consumes them. PUSH_STR validates exact constant index/length and embedded-NUL refusal. This intentionally conservative initial native boundary is separate from the broader private VM recipes.

Require entry arity/upvalues zero and exactly one INT result; initializer arity/upvalues zero and a native-supported result signature, preserving first named initializer ordering. Refuse STRUCT_NEW/GET/SET rather than silently translating them as AGG operations; only already-proved AGG_RECORD pack/get/set is in the first native route. Map each constructor's record ordinal to the copied declaration layout explicitly, checking tag, field count and nominal identity; require descriptor.layout==descriptor.ordinal for every copied record descriptor, including unused declarations, matching the existing public native guard. Nonidentity ordering refuses rather than being relabeled. Canonical private output enables map runtime declarations even if no HM opcode is present. Keep all existing local/shape/native output limits. Check every instruction and recipe, including unreachable code. An exact guard refusal at the four-original first gate stops qualification for review.

Parent approved this interface and source-only implementation checkpoint. No build or runtime equivalence is claimed here.
