# My multi-instance service transport

I use service-binding version 3 to retain multiple nominal File and TCP
catalog instances in one NanoISA v2 module. I preserve the existing version 1
and version 2 encodings. My new table describes metadata; it supplies no host
authority and does not admit mixed executable code yet. I track flow, dispatch,
source lowering and grant integration separately under #990.

## Wire table

I encode integers in little-endian order. My header is 16 bytes:

| Offset | Width | Meaning |
| --- | --- | --- |
| 0 | 2 | Service-binding version, exactly 3 |
| 2 | 2 | Reserved, zero |
| 4 | 4 | Instance count, 1 through 64 |
| 8 | 4 | Instance entry size, exactly 64 |
| 12 | 4 | Reserved, zero |

Each 64-byte entry has the following fields:

| Offset | Width | Meaning |
| --- | --- | --- |
| 0 | 4 | Instance ordinal, equal to its table position |
| 4 | 2 | Immutable catalog identity: 1 File, 2 TCP |
| 6 | 2 | Reserved, zero |
| 8 | 20 | Five import indices, in catalog method order |
| 28 | 36 | Nine layout slots, in catalog type order |

File has eight nominal types. I require its ninth slot to be `UINT32_MAX`.
TCP uses all nine slots, including Endpoint. Every active import index is
unique across all instances; every active layout index is likewise unique.
I forbid `UINT32_MAX` in an active slot. Import and layout indices belong to
separate tables and may have equal numeric values.

My exact extent is `16 + 64 * instance_count`, with no trailing bytes. The
largest table is 4,112 bytes. These are explicit implementation limits, not a
claim that every possible service graph fits. I refuse larger tables before
allocation. I do not allocate during raw encoding or decoding. Encoding uses
a bounded staging buffer so input and byte output may overlap; decoding uses
a staged value. Failure preserves output buffers and the caller's size value.
The size pointer must be disjoint from input and output storage.

## Nominal identity

An instance ordinal distinguishes repeated declarations of the same catalog.
For each instance I require its exact five methods, signatures, named layouts,
member tags and prior-only layout edges. A member's nominal target must belong
to that instance. A same-shaped File in another instance cannot be substituted
into its OpenResult. Layout and import table order may differ from catalog
order; the explicit indices retain identity through either order.

I validate ownership-v1 flags and function descriptors against the complete
combined map. Each instance's owner and owned-result layouts carry the exact
resource flag; borrowed parameters identify that instance's owner. Unmapped
layouts retain no catalog authority. My query returns instance, catalog,
catalog ordinal, global layout index and per-kind source ordinal separately.
The returned plan owns its metadata and remains valid after the input module
is freed. I retain the 65,536-layout bound from the single-catalog validators.

## Module boundary

I validate the complete map before retaining service ownership through either
module bridge. Serialization and deserialization preserve service, layout and
ownership bytes. Module feature bits and ordinary structural validation still
apply. An unknown or malformed version cannot fall back to a different codec.

My general verifier, VM and native translator still refuse service execution.
Existing explicitly granted File and TCP execution remains single-catalog;
this table does not combine their grants. My checked mixed-flow queries are described below. Runtime values,
cleanup, paired source lowering and supervised publication remain necessary
before a mixed source program can execute.

I test mixed File/TCP/File modules, repeated-catalog identity, import/layout
permutations, cross-instance payload substitution, exact ownership, every raw
truncation at the maximum extent, single-byte mutations, overlap, malformed
serialized metadata with corrected CRCs, allocation failure and both bridge
directions with `make -f Makefile.gnu test-multi-nominal`. Sanitizers cover the
new codec/query and transport adapters; linked surrounding objects retain
their ordinary build flags. This is not a fully instrumented whole-program run.

## My checked mixed-flow queries

I instantiate the existing ownership, CODE, body, cyclic and indirect engines
with `services_flow_config.h`. File and TCP retain their original catalog
configuration. The mixed configuration resolves every member and method through
the nominal instance map; it does not search all instances for a matching name.

My private flow API uses `instance * 9 + type` for type identities and
`instance * 5 + method` for method identities. A File instance's unused ninth
type slot remains invalid. These identifiers differ from the global layout or
import indices, the wire catalog IDs, and the per-kind source ordinals used by
aggregate instructions. I retain all of those distinctions in the query facts.
The historical `catalog_ordinal` field name in the shared report shape carries
this private instance-qualified identifier in `NvmServices*` reports.

Calls and borrows compare exact declarations, including their global nominal
index. Acquiring and consuming services use the selected instance's owner and
Result. Result payloads and constructors resolve within that instance. TCP
begin-connect keeps its Endpoint-domain obligation; every byte-write method
keeps its byte-domain obligation. Host rights, liveness and cleanup obligations
remain pending. A logical call or hosted query never discharges host authority.

The hosted preflight bounds all declared imports, including bridge-array growth
and parameter storage for up to 320 imports. I preserve the existing function,
state, graph and allocation limits. The copied hosted plan owns its input bytes
and metadata; it remains usable after the input module and buffer are destroyed.

`make -f Makefile.gnu test-services-flow` checks mixed File/TCP/File programs
through direct and indirect owned/borrowed calls, all catalog methods, Ok/Error
branches, loops and permuted maps. I also check a full 64-instance table,
wrong-instance calls, service operands, result storage and error constructors,
failed-transition preservation, query lifetime and allocation failures. These
are synthetic logical programs; Endpoint values still require runtime domain
checking. My mixed runtime, grants and paired source product remain open.

## Private mixed value lifetime

I retain one independent File or TCP lifetime core per instance in
`src/nsi_services_values.{h,c}`. My trusted, serialized host caller supplies a
copied catalog table of at most 64 entries. This table alone is not a checked
module, a host grant, or permission to publish source execution.

My value and borrow carriers retain a one-based instance identity, a catalog
tag, and the corresponding core identity. I check the target instance before
calling a service. The core still checks invocation, generation, ownership and
borrow epochs. I retain File and TCP result representations separately; their
status numbers, cleanup fields and scalar result kinds are not interchangeable.
Repeated File instances have independent capacity and lifetime state.

I stage result publication after accepted core operations. A failed checked
operation preserves caller outputs; an accepted host error remains a typed
Result. Move and take-Ok consume their source, close consumes either host Result
arm, and a live borrow prevents ordinary move/drop/close. Terminal cleanup
reclaims every instance, including abandoned borrows and unhandled Results.
I cache the first finish report and preserve per-instance cleanup records.
The aggregate cleanup count counts those records, not distinct host closes:
TCP retains both a close error and its terminal ambiguous-close history.

My [mixed lifetime evidence](evidence/mixed-service-values-20261010/README.md)
qualifies this private carrier. My next runtime integration must derive the
catalog table from the retained checked nominal plan, preserve instance IDs
through frames and VM/native calls, enforce host grants before acquisition, and
publish scalar results only after clean cleanup. This carrier does not establish
those dispatch, source, grant or release requirements.
